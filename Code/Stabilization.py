import numpy as np
import cv2


SMOOTHING_RADIUS = 50
fourcc = cv2.VideoWriter_fourcc(*'XVID')

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory


def fixBorder(frame, unstablize = False):
    s = frame.shape
    # Scale the image 4% without moving the center
    if(unstablize):
        scaleFactor = 1.04
    else:
        scaleFactor = 25/26
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, scaleFactor)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


# Read input video

def stablize(video):
    count = 0
    old_p = 0
    cp = cv2.VideoCapture(video)

    # To get number of frames
    n_frames = int(cp.get(cv2.CAP_PROP_FRAME_COUNT))

    # To check the number of frames in the video
    width = int(cp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cp.get(cv2.CAP_PROP_FRAME_HEIGHT))


    # get the number of frames per second
    fps = cp.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # Try doing 2*width
    out_reg = cv2.VideoWriter('../Outputs/stabilize.avi', fourcc, fps, (width, height), 1)
    # read the first frame
    _, prev = cp.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 2):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

        succ, curr = cp.read()

        if not succ:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Track feature points
        # status = 1. if flow points are found
        # err if flow was not find the error is not defined
        # curr_pts = calculated new positions of input features in the second image
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        assert prev_pts.shape == curr_pts.shape

        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]

        dx = m[0, 2]
        dy = m[1, 2]

        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [dx, dy, da]

        prev_gray = curr_gray


    # Find the cumulative sum of tranform matrix for each dx,dy and da
    trajectory = np.cumsum(transforms, axis=0)

    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    cp.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Write n_frames-1 transformed frames
    for i in range(n_frames - 2):
        # Read next frame
        success, frame = cp.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (width, height))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # Write the frame to the file
        frame_out = cv2.hconcat([frame, frame_stabilized])

        # If the image is too big, resize it.
        if (frame_out.shape[1] > 1920):
            frame_out = cv2.resize(frame_out, (int(frame_out.shape[1] / 2), int(frame_out.shape[0] / 2)));

        frame_out = frame_out.astype('uint8')
        frame_stabilized = frame_stabilized.astype('uint8')
        out_reg.write(frame_stabilized)
        count+=1
        new_p = count * 100 // n_frames
        if (new_p != old_p):
            print(str(new_p) + "% completed")
        old_p = new_p

    # Release video
    cp.release()
    out_reg.release()
    # Close windows
    cv2.destroyAllWindows()

def unstablize(analyze_name, target_name):
    out = "../Outputs/unstabilized_alpha.avi"
    cap = cv2.VideoCapture(analyze_name)
    target_cap = cv2.VideoCapture(target_name)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_reg = cv2.VideoWriter(out, fourcc, fps, (width, height),1)
    # read the first frame
    _, prev = cap.read()

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 2):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=50,
                                           qualityLevel=0.1,
                                           minDistance=40,
                                           blockSize=40,
                                           useHarrisDetector=False)

        succ, curr = cap.read()

        if not succ:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        assert prev_pts.shape == curr_pts.shape

        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]

        dx = m[0, 2]
        dy = m[1, 2]

        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [dx, dy, da]

        prev_gray = curr_gray


    # Find the cumulative sum of tranform matrix for each dx,dy and da
    trajectory = np.cumsum(transforms, axis=0)

    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    for i in range(n_frames - 2):
        # Read next frame
        success, frame = target_cap.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        MULTIPLIER = 1.0
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = 1 #np.cos(da)
        m[0, 1] = np.sin(da)*MULTIPLIER
        m[1, 0] = -np.sin(da)*MULTIPLIER
        m[1, 1] = 1 #np.cos(da)
        m[0, 2] = -dx
        m[1, 2] = -dy



        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (width, height))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized, True)


        frame_stabilized = frame_stabilized.astype('uint8')
        out_reg.write(frame_stabilized)


    # Release video
    cap.release()
    out_reg.release()
    # Close windows
    cv2.destroyAllWindows()
