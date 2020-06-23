import cv2
import numpy as np
from scipy import ndimage

outputNameBG = "med12bg.avi"
outputNameFG = "med12fg.avi"
cap = cv2.VideoCapture("stabilize.avi")
panel = np.zeros([100, 700], np.uint8)
#cv2.namedWindow('panel')
fgbg = cv2.createBackgroundSubtractorMOG2()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out_size = (width, height)
outBG = cv2.VideoWriter(outputNameBG, fourcc, fps, out_size, 1)
outFG = cv2.VideoWriter(outputNameFG, fourcc, fps, out_size, 1)


def nothing(x):
    pass


while cap.isOpened():

    ret, frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([0, 0, 0])
        upper_green = np.array([120, 60, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(frame, frame, mask=mask)
        fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        foreground0 = (fg[:, :, 0])
        #foreground0 = ndimage.binary_fill_holes(foreground0, structure=np.ones((25,25))).astype(int)
        foreground0[np.abs(foreground0) == 1] = 255
        foreground1 = (fg[:, :, 1])
        #foreground1 = ndimage.binary_fill_holes(foreground0, structure=np.ones((25,25))).astype(int)
        foreground1[np.abs(foreground1) == 1] = 255
        foreground2 = (fg[:, :, 2])
        #foreground2 = ndimage.binary_fill_holes(foreground0, structure=np.ones((25,25))).astype(int)
        foreground2[np.abs(foreground2) == 1] = 255

        foreground0 = foreground0.astype('uint8')
        foreground1 = foreground1.astype('uint8')
        foreground2 = foreground2.astype('uint8')

        fg = np.dstack((foreground0, foreground1))
        fg = np.dstack((fg, foreground2))

        #cv2.imshow('bg', bg)
        #cv2.imshow('fg', fg)

        #cv2.imshow('panel', panel)
        outBG.write(bg)
        outFG.write(fg)
    else:
        break

cap.release()
cv2.destroyAllWindows()
