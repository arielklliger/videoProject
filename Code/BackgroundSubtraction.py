import cv2
import numpy as np



def BS(video):

    outputNameMask = "../Output/binary.avi"
    outputNameExtracted = "../Output/extracted.avi"
    cap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_size = (width, height)
    outMask = cv2.VideoWriter(outputNameMask, fourcc, fps, out_size, 0)
    outExtracted = cv2.VideoWriter(outputNameExtracted, fourcc, fps, out_size, 1)
    kernel = np.ones((5, 5), np.uint8)

    cap = cv2.VideoCapture(video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # MOGsubtractor = cv2.createBackgroundSubtractorMOG2(history=204, varThreshold=50, detectShadows=True)
    KNNsubtractor = cv2.createBackgroundSubtractorKNN(history=204, detectShadows=True)
    count = 0
    old_p = 0
    i = 0
    # feed the video into the subtractor
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)




            gray = frame

            KNNsubtractor.apply(gray)
            i = i + 1
        else:
            break

    # start creating binary
    mean = 0
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            # bg_difference = cv2.absdiff(rotem_curr, bg)

            #colors:
            # colors


            gray = frame
            KNNmask = KNNsubtractor.apply(gray)
            maskOR = KNNmask
            maskOR[np.abs(maskOR) < 255] = 0
            # maskOR = cv2.morphologyEx(maskOR, cv2.MORPH_OPEN, kernel)
            # maskOR = cv2.morphologyEx(maskOR, cv2.MORPH_CLOSE, kernel)

            # contours
            # maskOR = cv2.dilate(maskOR,kernel,iterations = 1)

            #maskOR = cv2.GaussianBlur(maskOR, (3, 3), 0)
            #maskOR = cv2.medianBlur(maskOR, 15)
            maskOR = cv2.dilate(maskOR, kernel, iterations=2)
            contours, _ = cv2.findContours(maskOR, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: len(x), reverse=True)
            mean = mean + contours[0].shape[0]
            singleContourMask = np.zeros((height, width), np.uint8)
            singleContourMask = cv2.drawContours(singleContourMask, contours, 0, (255, 255, 255), -1)

            maskOR = singleContourMask
            maskOR = cv2.erode(maskOR,kernel,iterations =2)
            maskOR = cv2.medianBlur(maskOR, 23)

            maskOR[maskOR < 100] = 0
            maskOR[maskOR > 100] = 255

            # make extracted video
            foreground0 = (frame[:, :, 0])
            foreground1 = (frame[:, :, 1])
            foreground2 = (frame[:, :, 2])
            mask0 = maskOR
            fg0 = (foreground0 * (mask0 / 255))
            fg0 = fg0.astype('uint8')
            fg1 = (foreground1 * (mask0 / 255))
            fg1 = fg1.astype('uint8')
            fg2 = (foreground2 * (mask0 / 255))
            fg2 = fg2.astype('uint8')
            fg = np.dstack((fg0, fg1))
            fg = np.dstack((fg, fg2))
            maskOR = maskOR.astype('uint8')
            outExtracted.write(fg)
            outMask.write(maskOR)
            count += 1
            new_p = count * 100 // n_frames
            if (new_p != old_p):
                print(str(new_p) + "% completed")
            old_p = new_p


        else:
            break
    cap.release()
    cv2.destroyAllWindows()