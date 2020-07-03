import numpy as np
import cv2




def bgs(video):
    cap = cv2.VideoCapture(video)
    outputNameMask = "../Outputs/binary.avi"
    outputNameExtracted = "../Outputs/extracted.avi"
    weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vidLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    iters = 2
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = n_frames *(1+iters)
    count = 0
    old_p = 0
    #video writers
    binary = cv2.VideoWriter(outputNameMask, fourcc, fps, (weight, height))
    extracted = cv2.VideoWriter(outputNameExtracted, fourcc, fps, (weight, height))
    KNNmask = cv2.createBackgroundSubtractorKNN(history=vidLength * (iters + 1), detectShadows=True)



    for i in range(iters):
        while (True):
            ret, frame = cap.read()
            if (not ret):
                break
            blurredFrame = addBlur(frame)
            KNNmask.apply(blurredFrame)
            count += 1
            new_p = count * 100 // n_frames
            if (new_p != old_p):
                print(str(new_p) + "% completed")
            old_p = new_p
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    while(True):
        ret, frame = cap.read()
        if(not ret):
            break


        blurredFrame = addBlur(frame)
        KNNMask = KNNmask.apply(blurredFrame)
        bgsMask = KNNMask
        bgsMask = cv2.morphologyEx(bgsMask, cv2.MORPH_DILATE, (4, 4), iterations=3)
        _, fullMask = cv2.threshold(bgsMask, 128, 255, cv2.THRESH_BINARY)

        #limit hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = addBlur(hsv)
        low = np.array([85, 50, 50])
        high = np.array([130, 255, 255])
        v_channel_mask = cv2.inRange(hsv, low, high)
        v_channel_mask = cv2.bitwise_not(v_channel_mask)
        fullMask = cv2.bitwise_or(v_channel_mask, v_channel_mask, mask=fullMask)
        low_s = np.array([0, 50, 100])
        high_s = np.array([10, 150, 255])
        sMask = cv2.inRange(hsv, low_s, high_s)
        sMask = cv2.morphologyEx(sMask, cv2.MORPH_ERODE, (10, 10), iterations=3)
        sMask = cv2.morphologyEx(sMask, cv2.MORPH_CLOSE, (10, 10), iterations=3)
        fullMask = cv2.bitwise_or(fullMask, sMask)

        fullMask = cv2.medianBlur(fullMask, 7)
        fullMask = cv2.medianBlur(fullMask, 5)


        contours, _ = cv2.findContours(fullMask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: len(x), reverse=True)
        singleContourMask = np.zeros((height, weight), np.uint8)
        singleContourMask = cv2.drawContours(singleContourMask, contours, 0, (255, 255, 255), -1)

        _, singleContourMask = cv2.threshold(singleContourMask, 0, 255, cv2.THRESH_OTSU)

        Mask = singleContourMask
        Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, (7, 7), iterations=3)
        Mask = cv2.morphologyEx(Mask, cv2.MORPH_ERODE, (5, 5), iterations=2)
        Mask = cv2.morphologyEx(Mask, cv2.MORPH_ERODE, (5, 5), iterations=2)


        binary_frame = np.ones((height, weight, 3)).astype(np.uint8)*255
        binary_frame = cv2.bitwise_or(binary_frame, binary_frame, mask=Mask)
        binary.write(binary_frame)


        extracted_frame = cv2.bitwise_or(frame, frame, mask=Mask)
        extracted.write(extracted_frame)

        count += 1
        new_p = count * 100 // n_frames
        if (new_p != old_p):
            print(str(new_p) + "% completed")
        old_p = new_p

    cap.release()
    binary.release()
    extracted.release()
    cv2.destroyAllWindows()

def addBlur(frame):

    frame = cv2.GaussianBlur(frame, (5, 5), 0.3)
    frame = cv2.GaussianBlur(frame, (5, 5), 0.3)
    frame = cv2.medianBlur(frame, 5)

    return frame
