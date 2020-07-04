import GeodisTK
import cv2
import numpy as np
from scipy.stats import gaussian_kde
from CODE.Stabilization import  unstablize
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def getMedianBGFromVideo(video):

    cap = cv2.VideoCapture(video)
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

    frames = []
    for fid in frameIdsStart:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(frame)

    earlyMedian = np.median(frames, axis=0).astype(dtype=np.uint8)

    frames = []
    for fid in frameIdsEnd:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(frame)

    endMedian = np.median(frames, axis=0).astype(dtype=np.uint8)

    bg = cv2.bitwise_and(earlyMedian, earlyMedian, mask=right_mask) + cv2.bitwise_and(endMedian, endMedian, mask=left_mask)
    cv2.imwrite('../Temp/bg_cut.jpg', bg)
    return bg



def kde_scipy(x, x_grid, **kwargs):
    kde = gaussian_kde(x, bw_method='silverman', **kwargs)
    return kde.evaluate(x_grid)


def calc_alpha(frame_v_channel, binary, background, fg_prob, bg_prob):
    binary[binary < 255] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_prob_map = fg_prob[frame_v_channel]
    bg_prob_map = bg_prob[frame_v_channel]

    fg_prob_map = fg_prob_map.astype('float32')
    bg_prob_map = bg_prob_map.astype('float32')

    fg_prob_map_grad = fg_prob_map
    bg_prob_map_grad = bg_prob_map

    f_binary = cv2.erode(binary, small_kernel, iterations=1)
    b_binary = cv2.dilate(binary, kernel, iterations=1)


    gtk_f = GeodisTK.geodesic2d_raster_scan(fg_prob_map_grad, f_binary, 1, 1)
    gtk_b = GeodisTK.geodesic2d_raster_scan(bg_prob_map_grad, 255 - b_binary, 1, 1)


    wf = np.multiply(fg_prob_map, np.power(gtk_f, -1.2))
    wb = np.multiply(bg_prob_map, np.power(gtk_b, -1.2))
    mask = np.zeros(binary.shape)
    mask[np.bitwise_or(binary != f_binary, binary != b_binary)] = 1

    binary[binary < 255] = 0
    binary[binary == 255] = 1

    alpha_map = np.divide(wf, wf + wb)
    where_are_NaNs = np.isnan(alpha_map)
    alpha_map[where_are_NaNs] = 1

    binary = binary.astype('float64')

    alpha = np.add(np.multiply(mask, alpha_map), np.multiply(1 - mask, binary))

    return alpha


def matting(stabilize, binary, background):
    stab_cap = cv2.VideoCapture(stabilize)
    bin_cap = cv2.VideoCapture(binary)
    background = cv2.imread(background, cv2.IMREAD_COLOR)
    old_bg = getMedianBGFromVideo(stabilize)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = stab_cap.get(cv2.CAP_PROP_FPS)
    height = int(stab_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(stab_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_size = (width, height)
    background = cv2.resize(background, (width, height))
    output_name = "../Outputs/matted.avi"
    output_writer = cv2.VideoWriter(output_name, fourcc, fps, out_size, 1)
    alpha_vid_name = "../Outputs/alpha.avi"
    alpha_vid = cv2.VideoWriter(alpha_vid_name, fourcc, fps, out_size, 0)

    count = 0
    old_p = 0
    n_frames = int(stab_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    onceBg = 0
    isFirst = 0

    #for tracking
    #bbox = (11, 400, 100, 800)
    #bbox = (0, 158, 517, 813)
    bbox = (31, 198, 444, 750)
    tracker = cv2.TrackerCSRT_create()

    while (stab_cap.isOpened() and bin_cap.isOpened()):
        init = np.zeros((height, width))
        stab_ret, stab_frame = stab_cap.read()
        bin_ret, bin_frame = bin_cap.read()
        if stab_ret and bin_ret:
            stab_frame_hsv = cv2.cvtColor(stab_frame, cv2.COLOR_RGB2HSV)
            bin_frame0 = bin_frame[:, :, 0]
            bin_frame0[bin_frame0 < 100] = 0
            bin_frame0[bin_frame0 > 100] = 255

            if (isFirst == 0):
                frame_v_channel = stab_frame_hsv[:, :, 2]
                background_HSV = cv2.cvtColor(old_bg, cv2.COLOR_RGB2HSV)
                old_bg_v = background_HSV[:, :, 2]

                # kde
                foreground_dataset = (frame_v_channel[bin_frame0 == 255]).T

                grid = np.linspace(0, 255, 256)
                foreground_pdf = kde_scipy(foreground_dataset, grid)
                ok = tracker.init(stab_frame, bbox)
                background_resize = cv2.resize(old_bg_v, (int(width / 2), int(height / 2)))
                background_dataset = background_resize.flatten()
                background_pdf = kde_scipy(background_dataset, grid)
                fg_prob = foreground_pdf / (foreground_pdf + background_pdf)
                bg_prob = background_pdf / (foreground_pdf + background_pdf)
                isFirst = 1

            #track
            ok, bbox = tracker.update(stab_frame)
            if not ok:
                print("Tracking in matting failed")
                return
            stab_frame_resize = stab_frame_hsv[int(bbox[1]):int(bbox[1])+int(bbox[3]),int(bbox[0]):int(bbox[0])+int(bbox[2]),:]
            bin_frame0 = bin_frame0[int(bbox[1]):int(bbox[1])+int(bbox[3]),int(bbox[0]):int(bbox[0])+int(bbox[2])]

            alpha = calc_alpha(stab_frame_resize[:, :, 2], bin_frame0, background, fg_prob, bg_prob)

            #track
            init[int(bbox[1]):int(bbox[1])+int(bbox[3]),int(bbox[0]):int(bbox[0])+int(bbox[2])] = alpha
            alpha = init

            mat_frame0 = np.add(np.multiply(alpha, stab_frame[:, :, 0]), np.multiply(1 - alpha, background[:, :, 0]))
            mat_frame1 = np.add(np.multiply(alpha, stab_frame[:, :, 1]), np.multiply(1 - alpha, background[:, :, 1]))
            mat_frame2 = np.add(np.multiply(alpha, stab_frame[:, :, 2]), np.multiply(1 - alpha, background[:, :, 2]))

            mat_frame = np.dstack((np.dstack((mat_frame0, mat_frame1)), mat_frame2))
            mat_frame = mat_frame.astype('uint8')
            alpha = alpha * 255
            alpha = alpha.astype('uint8')

            count += 1
            new_p = count * 100 // n_frames
            if (new_p != old_p):
                print(str(new_p) + "% completed")
            old_p = new_p
            output_writer.write(mat_frame)
            alpha_vid.write(alpha)

        else:
            break



    bin_cap.release()
    cv2.destroyAllWindows()
    stab_cap.release()
    cv2.destroyAllWindows()
    unstablize("../Input/INPUT.avi", "../Outputs/alpha.avi")




