import numpy as np
from cv2 import cv2
import matplotlib
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('Oseba1_D_take1.mp4')
fps=cap.get(cv2.CAP_PROP_FPS)
frameNum=0
c=0
hole=np.empty(9)
numOfCircles=0
previousFrame=None
hist_end = 0
start_time = 0
stop_time = 0
holesFull=0
holesEmpty=1
endgame=0

while True:   
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q') or ret==False or endgame==1:
        print("Exiting at frame number: "+str(frameNum))
        cap.release()
        cv2.destroyAllWindows()
        print((frameNum - start_time)/fps, 'seconds')
        print(fps, 'fps')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imHSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    _, threshold = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
    imHSV=cv2.resize(imHSV,(960,540))
    imHSV=imHSV[120:480,100:900]
    vid_res = cv2.resize(threshold, (960, 540))
    image=vid_res[120:480,100:900]
    image8=image.astype('uint8')
    vid_res_frame = cv2.resize(frame, (960, 540))

    if frameNum==0:
        circles=cv2.HoughCircles(image8,cv2.HOUGH_GRADIENT,1,10,param1=15,param2=15,minRadius=5,maxRadius=13)

        if circles is not None:
            for num in circles[0,:]:
                numOfCircles+=1

        if numOfCircles==9:
            print("Number of recognized holes is: "+str(numOfCircles))
        elif numOfCircles<9:
            print("Not enough holes recognized!")
        elif numOfCircles>9:
            print("Too much holes recognized!")

    if numOfCircles==9:
        for i in range(9):
            r=round(circles[0,i,2])-2
            d=round(2*circles[0,i,2])-4
            x=round(circles[0,i,0]-r)
            y=round(circles[0,i,1]-r)
            hole[i]=imHSV[y:y+d,x:x+d,2].mean()

    if holesFull!=1 and holesEmpty==1:
        if((hole[0]>92)&(hole[1]>92)&(hole[2]>92)&(hole[3]>92)&(hole[4]>92)&(hole[5]>92)&(hole[6]>92)&(hole[7]>92)&(hole[8]>92)):
            holesFull=1
            holesEmpty=0

    if holesEmpty==0:
        if((hole[0]<85)&(hole[1]<85)&(hole[2]<85)&(hole[3]<85)&(hole[4]<85)&(hole[5]<85)&(hole[6]<85)&(hole[7]<85)&(hole[8]<85)):
            endgame=1

    frameNum+=1
    cv2.imshow('frame', vid_res_frame)

    if c%5==0:

        if ((previousFrame is not None) and (hist_end==0)):
            H1 = cv2.calcHist([image8],[0],None,[256],[0,256])
            H2 = cv2.calcHist([previousFrame],[0],None,[256],[0,256])

            compareHist = cv2.compareHist(H1, H2, cv2.HISTCMP_HELLINGER)

            if compareHist>=0.01:
                start_time = frameNum
                hist_end = 1

        plot = plt.show()
        previousFrame = image8
    c+=1