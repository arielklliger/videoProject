from videoProject.Code.Stabilization import stablize
from videoProject.Code.BackgroundSubtraction import BS
from videoProject.Code.Matting import matting
from videoProject.Code.Tracking import track
import time




def runme():
    #start logging
    f = open("../Output/RunTimeLog.txt", "a")
    start_time = time.time()
    print("Start")
    #start stablize

    print("Start stablize")
    stablize("../Input/INPUT.avi")
    print("Done stablize")
    t = time.time() - start_time
    print("--- %s seconds ---" % (t))
    f.write("Stablize took "+ str(t)+" seconds"+'\n')

    #start BS
    print("Start background subtraction")
    t = time.time()
    BS("../Output/stabilize.avi")
    print("Done background subtraction")
    t = time.time() - t
    print("--- %s seconds ---" % (t))
    f.write("background took " + str(t) + " seconds."+'\n')

    #start matting
    t = time.time()
    print("Start matting")
    matting("../Output/stabilize.avi","../Output/binary.avi",'../Input/background.jpg')
    print("Done matting")
    t = time.time() - t
    print("--- %s seconds ---" % (t))
    f.write("matting took " + str(t) + " seconds."+'\n')

   #start tracking
    t = time.time()
    print("Start tracking")
    track("../Output/matted.avi")
    print("Done tracking")
    t = time.time() - t
    print("--- %s seconds ---" % (t))
    f.write("tracking took " + str(t) + " seconds"+'\n')

    #end
    total = time.time()-start_time
    print("Done")
    f.write("Total run took " + str(total)+ " seconds")

#runme()
BS("../Output/med12fg.avi")
#matting("../Output/stabilize.avi","../Output/binary_contour_or.avi",'../Input/background.jpg')