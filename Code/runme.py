from CODE.Stabilization import stablize
from CODE.Matting import matting
from CODE.BackgroundSubtractor import bgs
from CODE.Tracking import track
from CODE.Stabilization import unstablize
import time


def runme():
    #start logging
    f = open("../Outputs/RunTimeLog.txt", "a")
    start_time = time.time()
    print("################")
    print("Start Project")
    print("################")

    #start stablize
    print("Start Stabilization")
    stablize("../Input/INPUT.avi")
    print("Done Stabilization")
    t = time.time() - start_time
    print("Stabilization took {:.1f} seconds".format(t))
    f.write("Stabilization took {:.1f} seconds \n".format(t))
    print("################")

    #start BS
    print("Start background subtraction")
    t = time.time()
    bgs("../Outputs/stabilize.avi")
    print("Done background subtraction")
    t = time.time() - t
    print("Background Subtraction took {:.1f} seconds".format(t))
    f.write("Background Subtraction took {:.1f} seconds \n".format(t))

    #start matting
    t = time.time()
    print("################")
    print("Start Matting")
    matting("../Outputs/stabilize.avi", "../Outputs/binary.avi", '../Input/background.jpg')
    print("Done Matting")
    t = time.time() - t
    print("Matting took {:.1f} seconds".format(t))
    f.write("Matting took {:.1f} seconds \n".format(t))

   #start tracking
    t = time.time()
    print("################")
    print("Start Tracking")
    track("../Outputs/matted.avi")
    print("Done Tracking")
    t = time.time() - t
    print("Tracking took {:.1f} seconds".format(t))
    f.write("Tracking took {:.1f} seconds \n".format(t))

    #end
    total = time.time()-start_time
    print("################")
    print("Done Project")
    print("Total run took {:.1f} seconds".format(total))
    print("################")
    f.write("Total run took {:.1f} seconds \n".format(total))


runme()

###########################
# RUN BLOCKS INDIVIDUALLY #
###########################
# For running stabilization only un-comment this line:
# stablize("../Input/INPUT.avi")

# For running BS only un-comment this line:
#bgs("../Outputs/stabilize.avi")

# For running matting only un-comment this line:
#matting("../Outputs/stabilize.avi", "../Outputs/binary.avi", '../Input/background.jpg')

# For running tracking only un-comment this line:
# If you want to choose the tracking box change this user_input flag to True
#user_input = False
#track("../Outputs/matted.avi", user_input)
##########
