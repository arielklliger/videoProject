import cv2


def track(video,userInput=False):
    outputNameFG = "../Outputs/OUTPUT.avi"
    tracker = cv2.TrackerCSRT_create()
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
    if not video.isOpened():
        print("Could not open video")
        return


    ok, frame = video.read()

    bbox = (0, 210, 405, 777)
    if (userInput):
        bbox = cv2.selectROI(frame, False)

    ok = tracker.init(frame, bbox)

    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break

        ok, bbox = tracker.update(frame)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        frame = frame.astype('uint8')
        outFG.write(frame)
        count +=1
        new_p = count * 100 // n_frames
        if (new_p != old_p):
            print(str(new_p) + "% completed")
        old_p = new_p

    video.release()
    cv2.destroyAllWindows()