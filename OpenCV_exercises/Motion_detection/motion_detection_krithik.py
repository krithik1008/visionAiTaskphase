
#Select level from trackbar 
import cv2 as cv
import numpy as np

cap=cv.VideoCapture(0)

def nochange(x):    #function for tacknar parameter when tracker changes
    pass

cv.namedWindow('Contour Area')
cv.createTrackbar('Level','Contour Area',0,3,nochange)
cv.resizeWindow('Contour Area',600,100)
#Trackbar level (contour area) for size of moving object:
#0 -> 100
#1 -> 500
#2 -> 900
#3 -> 1300
area=np.array([100,500,900,1300])  #assigning area to each level of trackbar
color=np.array([[0,255,0],[255,255,0],[255,0,255],[0,255,255]])#diff color for each level
while True:
    pos=cv.getTrackbarPos('Level','Contour Area')
    #capturing consecutive frames for processing
    _,frame1=cap.read()
    _,frame2=cap.read()
    
    #porcessing difference in 2 frames to detect motion
    d=cv.absdiff(frame1,frame2)
    gray=cv.cvtColor(d,cv.COLOR_BGR2GRAY)
    b=cv.GaussianBlur(gray,ksize=(5,5),sigmaX=0)
    _, thresh=cv.threshold(b,20,255,cv.THRESH_BINARY)
    dilated=cv.dilate(thresh,None,iterations=3)
    cv.imshow("d",dilated)
    
    contours,_=cv.findContours(dilated,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        cv.putText(frame1, "Level: {}".format(pos), (340, 20), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)
        if cv.contourArea(contour) < area[pos]:
            continue
        cv.rectangle(frame1, (x, y), (x+w, y+h), (int(color[pos][0]),int(color[pos][1]), int(color[pos][2])),2)
        cv.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)
    image = cv.resize(frame1, (1280,720))
    cv.imshow("img", image)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv.waitKey(40) == 27: # Esc to quit the video windows
        break

cv.destroyAllWindows()
cap.release()   

