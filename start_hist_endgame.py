import numpy as np
from cv2 import cv2
import matplotlib
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('Oseba1_D_take1.mp4')
fps=cap.get(cv2.CAP_PROP_FPS)
frameNum=0
hole=np.empty(9)
numOfCircles=0
previousFrame=None
hist_end = 0
start_time = 0
stop_time = 0
holesFull=0
holesEmpty=1
endgame=0
play=1

while True:   
    ret, frame = cap.read()

    if endgame==1 and play==1:
        print('Number of frames: '+str(frameNum-start_time))
        print(fps, 'fps')
        print((frameNum - start_time)/fps, 'seconds')
        play=0

    if cv2.waitKey(play) & 0xFF == ord('q') or ret==False:
        cap.release()
        cv2.destroyAllWindows()
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

        if numOfCircles<9:
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
        if(all(i>92 for i in hole)):
            holesFull=1
            holesEmpty=0

    if holesEmpty==0:
        if(all(i<90 for i in hole)):
            endgame=1

    frameNum+=1
    
    if start_time>0:
        cv2.putText(vid_res_frame,str(format((frameNum-start_time)/fps,'.2f'))+"s",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,150,150),2,cv2.LINE_AA)

    if (frameNum+1)%5==0:

        if ((previousFrame is not None) and (hist_end==0)):
            H1 = cv2.calcHist([image8],[0],None,[256],[0,256])
            H2 = cv2.calcHist([previousFrame],[0],None,[256],[0,256])

            compareHist = cv2.compareHist(H1, H2, cv2.HISTCMP_HELLINGER)

            if compareHist>=0.01:
                start_time = frameNum
                hist_end = 1

        previousFrame = image8

    cv2.imshow('frame', vid_res_frame)