import sys

import GeodisTK
import cv2
import numpy as np
from scipy.stats import gaussian_kde

import time

def getMedianBGFromVideo(video):

    # Open Video
    cap = cv2.VideoCapture(video)


    #Randomly select 25 frames
    #frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=150)

    frameIdsStart = [0, 1, 2]
    vidLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameIdsEnd = [vidLength-3, vidLength-2, vidLength-1]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    half_ones = np.ones((height, width//2)).astype(dtype=np.uint8)
    left_mask = np.zeros((height, width)).astype(dtype=np.uint8)
    left_mask[:, :width//2] = half_ones
    right_mask = np.zeros((height, width)).astype(dtype=np.uint8)
    right_mask[:, width//2:] = half_ones

    # Store selected frames in an array
    frames = []
    for fid in frameIdsStart:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(frame)

    medianStartFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

    frames = []
    for fid in frameIdsEnd:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(frame)

    medianEndFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

    # Display median frame
    # cv2.imshow('frame', medianStartFrame)
    # cv2.waitKey(0)
    # cv2.imshow('frame', medianEndFrame)
    # cv2.waitKey(0)
    bg = cv2.bitwise_and(medianStartFrame, medianStartFrame, mask=right_mask) + cv2.bitwise_and(medianEndFrame, medianEndFrame, mask=left_mask)
    cv2.imwrite('../Temp/bg_cut.jpg', bg)
    # cv2.waitKey(0)
    return bg


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)



    return new_frame
def BetterBuildFrame(frame,foreground_0_pdf,background_0_pdf,width,height):

    new_frame = np.zeros((height, width))
    sValues = np.array(frame).flatten()
    fgProbs = foreground_0_pdf[sValues]
    bgProbs = background_0_pdf[sValues]
    diff = np.array(fgProbs - bgProbs).reshape((height, width))
    new_frame[diff > 0] = 255
    return new_frame


    cv2.imwrite("kde_frame_0.jpg", new_frame)

    return new_frame
def BackS(video):
    fginitial = cv2.imread("../Temp/fgInitial.png")
    fginitial = cv2.cvtColor(fginitial, cv2.COLOR_BGR2GRAY)
    background = getMedianBGFromVideo(video)
    cap = cv2.VideoCapture(video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_size = (width, height)
    output_name = "../Outputs/binary.avi"
    outputNameExtracted = "../Outputs/extracted.avi"
    outExtracted = cv2.VideoWriter(outputNameExtracted, fourcc, fps, out_size, 1)
    bs_kde = cv2.VideoWriter(output_name, fourcc, fps, out_size, 0)
    isFirst = 1
    kernel = np.ones((10, 10), np.uint8)
    count =0
    old_p = 0
    #background = cv2.resize(background, (width, height))
    while (cap.isOpened()):
        ret,frame = cap.read()


        # bin_frame0 = bin_frame[:, :, 0]
        # bin_frame0[bin_frame0 < 100] = 0
        # bin_frame0[bin_frame0 > 100] = 255
        if ret:

            #using s channel
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            frame_0_channel = frame_hsv[:, :, 1]
            background_HSV = cv2.cvtColor(background, cv2.COLOR_RGB2HSV)
            background_0_channel = background_HSV[:, :, 1]

            #using quant and base
            frame_quant = frame/8
            frame_quant = frame_quant.astype('uint32')
            frame_quant = frame_quant[:,:,0] + frame_quant[:,:,1]*32
            background_quant = background/8
            background_quant =background_quant.astype('uint32')
            background_quant =background_quant[:, :, 0] +background_quant[:, :, 1] * 32

            frame_0_channel = frame_quant
            background_0_channel = background_quant

            if isFirst == 1:
                print("Preforming KDE, this will take a minute")
                cv2.imwrite("first_frame.jpg",frame)
                diff = frame - background
                cv2.imwrite("diff.jpg", diff)
                foreground_0_dataset = (frame_0_channel[fginitial == 0]).T
                background_0_dataset = (background_0_channel[fginitial == 255]).T
                # foreground_0_dataset = (frame_0_channel[fginitial == 0]).T
                # foreground_1_dataset = (frame_1_channel[fginitial == 0]).T
                # foreground_2_dataset = (frame_2_channel[fginitial == 0]).T
                # background_0_dataset = (background_0_channel[fginitial == 255]).T
                # background_1_dataset = (background_1_channel[fginitial == 255]).T
                # background_2_dataset = (background_2_channel[fginitial == 255]).T

                grid = np.linspace(0, 1023, 1024)
                foreground_0_pdf = kde_scipy(foreground_0_dataset, grid)
                # foreground_1_pdf = kde_scipy(foreground_1_dataset, grid)
                # foreground_2_pdf = kde_scipy(foreground_2_dataset, grid)
                background_0_pdf = kde_scipy(background_0_dataset, grid)
                # background_1_pdf = kde_scipy(background_1_dataset, grid)
                # background_2_pdf = kde_scipy(background_2_dataset, grid)

                fg_prob = foreground_0_pdf / (foreground_0_pdf + background_0_pdf)
                bg_prob = background_0_pdf / (foreground_0_pdf + background_0_pdf)
                isFirst =0
                print("Done preforming KDE")

            frame_bin = BetterBuildFrame(frame_0_channel,fg_prob,bg_prob,width,height)
            frame_bin = frame_bin.astype('uint8')
            frame_bin = cv2.morphologyEx(frame_bin, cv2.MORPH_OPEN, kernel)
            #frame_bin = cv2.morphologyEx(frame_bin, cv2.MORPH_CLOSE, kernel)


            #contours
            contours, _ = cv2.findContours(frame_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: len(x), reverse=True)
            singleContourMask = np.zeros((height, width), np.uint8)
            singleContourMask = cv2.drawContours(singleContourMask, contours, 0, (255, 255, 255), -1)

            frame_bin = singleContourMask
            frame_bin = cv2.morphologyEx(frame_bin, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite("first_kde.jpg", frame_bin)
            frame_bin = cv2.medianBlur(frame_bin, 23)

            #frame_bin = cv2.erode(frame_bin, kernel, iterations=2)
            #frame_v_channel[fg_prob>bg_prob] = 255
            #frame_v_channel[fg_prob<=bg_prob] = 0

            #cv2.imwrite("kde_frame.jpg",frame)
            frame_bin = frame_bin.astype('uint8')
            bs_kde.write(frame_bin)


            # make extracted video
            foreground0 = (frame[:, :, 0])
            foreground1 = (frame[:, :, 1])
            foreground2 = (frame[:, :, 2])
            mask0 = frame_bin
            fg0 = (foreground0 * (mask0 / 255))
            fg0 = fg0.astype('uint8')
            fg1 = (foreground1 * (mask0 / 255))
            fg1 = fg1.astype('uint8')
            fg2 = (foreground2 * (mask0 / 255))
            fg2 = fg2.astype('uint8')
            fg = np.dstack((fg0, fg1))
            fg = np.dstack((fg, fg2))
            outExtracted.write(fg)
            count += 1
            new_p = count * 100 // n_frames
            if (new_p != old_p):
                print(str(new_p) + "% completed")
            old_p = new_p


        else:
            break
    cap.release()
    cv2.destroyAllWindows()
# start_time = time.time()
# #BackS("../Outputs/stabilize.avi")
# BackS("../Outputs/Stabilized_Example_INPUT.avi")
# end_time = time.time()
# total = end_time-start_time
# print("took "+str(total)+" seconds.")

