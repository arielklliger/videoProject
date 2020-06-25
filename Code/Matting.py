import GeodisTK
import cv2
import numpy as np
from scipy.stats import gaussian_kde

bg = cv2.imread("background.jpg")


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)

def calc_alpha(frame_v_channel, binary, fginitial, background,fg_prob,bg_prob):
    binary[binary < 255] = 0 #######
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation = cv2.dilate(binary, kernel, iterations=1)
    frame_idx = np.indices(frame_v_channel.shape)
    narrowband_idx = frame_idx[:,binary != dilation]
    narrowband = frame_v_channel[binary != dilation]
    background_HSV = cv2.cvtColor(background, cv2.COLOR_RGB2HSV)
    background_v = background_HSV[:, :, 2]
    fg_prob_map = fg_prob[frame_v_channel]
    bg_prob_map = bg_prob[frame_v_channel]

    gtk_f = GeodisTK.geodesic2d_raster_scan(frame_v_channel, binary, 0.5, 5)
    gtk_b = GeodisTK.geodesic2d_raster_scan(background_v, binary, 0.5, 5)

    wf = fg_prob_map / gtk_f
    wb = bg_prob_map / gtk_b
    """
    trimap = binary
    trimap[trimap==255] = 1
    trimap[np.bitwise_or(binary != dilation, binary != erode)] = 2
    test = frame[trimap == 2]
    test = test.T
    pdf = kde_obj.evaluate(test)
    print(trimap.shape[0]-1)
    for i in range(1079):
        for j in range(trimap.shape[1]-1):
            if trimap[i,j] == 2:
                for m in range(pdf.shape[0]):
                    if(test[0,m] == frame[i,j,0] and test[1,m] == frame[i,j,1] and test[2,m] == frame[i,j,2]):
                        trimap[i, j] = pdf[m]
    """
    mask = np.zeros(binary.shape)
    mask[binary != dilation] = 1
    #mask = mask * 255
    #mask = mask.astype('uint8')
    cv2.imwrite("mask.jpg", mask)
    binary[binary<255] = 0
    binary[binary == 255] =1

    alpha = np.divide(wf, wf+wb)
    where_are_NaNs = np.isnan(alpha)
    alpha[where_are_NaNs] = 0

    #alpha[binary!=dilation] = 100
    #cv2.imwrite("alpha.jpg",alpha)
    #trimap[binary != dilation] = alpha[binary != dilation]
    #trimap[binary == dilation] = binary[[binary == dilation]]
    binary = binary.astype('float64')

    trimap = np.add(np.multiply(mask, alpha), np.multiply(1 - mask, binary))

    #trimap = trimap.astype('float64', """a, """ b)
    """
    trimap = trimap * 255
    trimap = trimap.astype('uint8')
    cv2.imwrite("trimap.jpg",trimap)
    plt.imshow(, cmap='gray')
    plt.show()
    """

    return trimap
def calc_alpha_new(frame, binary):
    binary[binary < 255] = 0
    binary = binary.astype('uint8')
    gtk = GeodisTK.geodesic2d_raster_scan(frame, binary, 0.5, 5)
    print(gtk)

def matting(stabilize, binary, background, fginitial):
    background = cv2.imread(background, cv2.IMREAD_COLOR)
    stab_cap = cv2.VideoCapture(stabilize)
    bin_cap = cv2.VideoCapture(binary)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = stab_cap.get(cv2.CAP_PROP_FPS)
    height = int(stab_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(stab_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_size = (width, height)
    background = cv2.resize(background, (width, height))
    output_name = "matted2406.avi"
    output_writer = cv2.VideoWriter(output_name, fourcc, fps, out_size, 1)
    alpha_vid_name = "alpha.avi"
    alpha_vid = cv2.VideoWriter(alpha_vid_name, fourcc, fps, out_size, 0)
    i = 0
    isFirst = 0
    while (stab_cap.isOpened() and bin_cap.isOpened()):
        stab_ret, stab_frame = stab_cap.read()
        stab_frame_hsv = cv2.cvtColor(stab_frame, cv2.COLOR_RGB2HSV)
        bin_ret, bin_frame = bin_cap.read()
        cv2.imwrite("bin.jpg", bin_frame)
        bin_frame0 = bin_frame[:,:,0]
        bin_frame0[bin_frame0<100] = 0
        bin_frame0[bin_frame0 > 100] = 255
        print(bin_frame.max())
        if stab_ret and bin_ret:

            if(isFirst==0):
                frame_v_channel = stab_frame_hsv[:,:,2]
                binary_inv = cv2.bitwise_not(bin_frame0)
                background_HSV = cv2.cvtColor(background, cv2.COLOR_RGB2HSV)
                background_v = background_HSV[:, :, 2]
                # kde
                foreground_dataset = (frame_v_channel[fginitial == 0]).T
                background_dataset = (frame_v_channel[fginitial == 255]).T

                f_dmax = foreground_dataset.max()
                f_dmin = foreground_dataset.min()
                b_max = background_dataset.max()
                b_min = background_dataset.min()
                grid = np.linspace(0, 255, 256)
                # x_grid = np.linspace(dmin, dmax, (dmax-dmin)+1)
                foreground_pdf = kde_scipy(foreground_dataset, grid)
                a = foreground_pdf.sum()
                background_pdf = kde_scipy(background_dataset, grid)

                fg_prob = foreground_pdf / (foreground_pdf + background_pdf)
                bg_prob = background_pdf / (foreground_pdf + background_pdf)
                isFirst = 1
            alpha = calc_alpha(stab_frame_hsv[:,:,2], bin_frame0, fginitial, background,fg_prob,bg_prob)

            mat_frame0 = np.add(np.multiply(alpha, stab_frame[:,:,0]), np.multiply(1-alpha,background[:,:,0]))
            mat_frame1 = np.add(np.multiply(alpha, stab_frame[:,:,1]), np.multiply(1-alpha,background[:,:,1]))
            mat_frame2 = np.add(np.multiply(alpha, stab_frame[:,:,2]), np.multiply(1-alpha,background[:,:,2]))

            mat_frame = np.dstack((np.dstack((mat_frame0, mat_frame1)),mat_frame2))
            alpha = alpha *255
            alpha = alpha.astype('uint8')

            cv2.imwrite("Frame " + str(i) + ".jpg", mat_frame)
            output_writer.write(mat_frame)
            alpha_vid.write(alpha)
            i=i+1
        else:
            break

    bin_cap.release()
    cv2.destroyAllWindows()
    stab_cap.release()
    cv2.destroyAllWindows()
    #makeMatted()




def makeMatted(video,mask,image):
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

