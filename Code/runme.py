from CODE.Stabilization import stablize
from CODE.Matting import matting
from CODE.BackgroundSubtractor import bgs
from CODE.Tracking import track
from CODE.Stabilization import  unstablize
import time


def runme():
    #start logging
    f = open("../Outputs/RunTimeLog.txt", "a")
    start_time = time.time()
    print("Start")
    #start stablize

    print("Start stablize")
    stablize("../Input/INPUT.avi")
    print("Done stablize")
    t = time.time() - start_time
    print("Stablize took "+ str(t)+" seconds"+'\n')
    f.write("Stablize took "+ str(t)+" seconds"+'\n')

    #start BS
    print("Start background subtraction")
    t = time.time()
    bgs("../Outputs/stabilize.avi")
    print("Done background subtraction")
    t = time.time() - t
    print("background subtraction took " + str(t) + " seconds."+'\n')
    f.write("background subtraction took " + str(t) + " seconds."+'\n')

    #start matting
    t = time.time()
    print("Start matting")
    matting("../Outputs/stabilize.avi","../Outputs/binary.avi",'../Input/background.jpg')
    print("Done matting")
    t = time.time() - t
    print("matting took " + str(t) + " seconds."+'\n')
    f.write("matting took " + str(t) + " seconds."+'\n')

   #start tracking
    t = time.time()
    print("Start tracking")
    track("../Outputs/matted.avi")
    print("Done tracking")
    t = time.time() - t
    print("tracking took " + str(t) + " seconds"+'\n')
    f.write("tracking took " + str(t) + " seconds"+'\n')

    #end
    total = time.time()-start_time
    print("Done")
    f.write("Total run took " + str(total)+ " seconds")


runme()
