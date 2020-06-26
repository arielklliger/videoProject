from Code.Stabilization import stablize
from Code.BackgroundSubtraction import BS
from Code.Matting import matting,makeMatted,stabilize_vid
from Code.Tracking import track
from Code.BS import BackS
import time
import cv2



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
    BackS("../Output/Stabilized_Example_INPUT.avi")
    print("Done background subtraction")
    t = time.time() - t
    print("--- %s seconds ---" % (t))
    f.write("background took " + str(t) + " seconds."+'\n')

    #start matting
    t = time.time()
    print("Start matting")
    makeMatted("../Output/extracted.avi","../Output/binary.avi",'../Input/background.jpg')
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

runme()
#stablize("../Input/INPUT.avi")
#BS("../Output/Stabilized_Example_INPUT.avi")
#stablize("../Input/INPUT.avi")
#fginitial = cv2.imread("fgInitial.png")
#fginitial = cv2.cvtColor(fginitial, cv2.COLOR_BGR2GRAY)
#start_time = time.time()
#matting("../Output/stabilize.avi","../Output/binary.avi",'../Input/background.jpg',fginitial)
#end_time = time.time() - start_time
#print("matting took "+str(end_time)+" seconds.")