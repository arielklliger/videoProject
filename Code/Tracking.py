import cv2
import sys

def track(video,userInput=False):
    outputNameFG = "../Outputs/OUTPUT.avi"
    tracker = cv2.TrackerCSRT_create()


    # Read video
    video = cv2.VideoCapture(video)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_size = (width, height)
    outFG = cv2.VideoWriter(outputNameFG, fourcc, fps, out_size, 1)
    count=0
    old_p=0
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    bbox = (0, 210, 405, 777)

    # Uncomment the line below to select a different bounding box
    if (userInput):
        bbox = cv2.selectROI(frame, False)
    #print (bbox)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = video.get(cv2.CAP_PROP_FPS)
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display result
        frame = frame.astype('uint8')
        outFG.write(frame)
        count +=1
        new_p = count * 100 // n_frames
        if (new_p != old_p):
            print(str(new_p) + "% completed")
        old_p = new_p