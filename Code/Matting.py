import cv2
import numpy as np
from scipy import ndimage


def matting(video,mask,image):
    outputNameFG = "../Output/matted.avi"
    cap = cv2.VideoCapture(video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_mask = cv2.VideoCapture(mask)
    # cv2.namedWindow('panel')
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_size = (width, height)
    outFG = cv2.VideoWriter(outputNameFG, fourcc, fps, out_size, 1)
    background = cv2.imread(image, cv2.IMREAD_COLOR)
    count = 0
    old_p = 0
    inv = np.ones((1080, 1920)) * 255
    while cap.isOpened():

        new_img = np.zeros((height, width))
        ret, frame = cap.read()
        ret_mask, frame_mask = cap_mask.read()

        if ret and ret_mask:
            foreground0 = (frame[:, :, 0])
            foreground1 = (frame[:, :, 1])
            foreground2 = (frame[:, :, 2])
            mask0 = (frame_mask[:, :, 0])
            mask1 = (frame_mask[:, :, 1])
            mask2 = (frame_mask[:, :, 2])

            fg0 = (foreground0 * (mask0 / 255)) + (background[:, :, 0] * ((inv - mask0) / 255))
            fg0 = fg0.astype('uint8')
            fg1 = foreground1 * (mask1 / 255) + (background[:, :, 1] * ((inv - mask1) / 255))
            fg1 = fg1.astype('uint8')
            fg2 = foreground2 * (mask2 / 255) + (background[:, :, 1] * ((inv - mask2) / 255))
            fg2 = fg2.astype('uint8')
            fg = np.dstack((fg0, fg1))
            fg = np.dstack((fg, fg2))

            filename_fg = 'colorfg' + str(count) + '.jpg'
            # cv2.imwrite(filename_fg, fg)
            outFG.write(fg)
            count += 1
            new_p = count * 100 // n_frames
            if (new_p != old_p):
                print(str(new_p) + "% completed")
            old_p = new_p
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
