#for proper functioning ensure that the background is plain (ideally a wall)
#try moving hand forward or backward if right gesture is not recognized 
#fit your hand (upto wrist) within the rectangle(ROI) 

import cv2 as cv
import numpy as np
import math

cap=cv.VideoCapture(0)


def dist(p1,p2):
  return math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))

while True:
    
    _,frame=cap.read()
    frame=cv.flip(frame,1)
    #region of interest where gesture are identified
    cv.rectangle(frame,(340,100),(600,300),(0,255,0),0)
    roi=frame[100:300,340:600]
    
    #processing Roi
    roi_blur = cv.GaussianBlur(roi, (15, 15), 0)
    hsv_roi=cv.cvtColor(roi_blur, cv.COLOR_BGR2HSV)
    
    #range of skin color
    su= np.array([180,255,255], dtype=np.uint8)
    sl= np.array([0,60,0], dtype=np.uint8)

    #creating a mask which will remove all non skin colored region
    mask = cv.inRange(hsv_roi, sl, su)
    #masking roi with the mask
    res=cv.bitwise_and(roi,roi,mask=mask)
    
    #thresholding to get a grayscale image
    _, res = cv.threshold(res, 25, 255, cv.THRESH_BINARY)
    res=cv.GaussianBlur(res, (5, 5), 0)
    h,s,v1=cv.split(res)
    res=v1
    res=res.astype(np.uint8)
    
    
    #finding and drawing contours 
    contour,hierarchy= cv.findContours(res,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    max_ct=max(contour, key = lambda x: cv.contourArea(x))
    epsilon = 0.00001*cv.arcLength(max_ct,True)
    c= cv.approxPolyDP(max_ct,epsilon,True)
    cv.drawContours(roi, c, -1, (255, 0, 255), 2)
    
    #finding extreme points of the contour
    l = tuple(c[c[:, :, 0].argmin()][0])
    r = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    
    #centroid of the contour
    M = cv.moments(max_ct)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cntr=(cx,cy)
    
    #classification of gestures based on dist between points of contour
    font = cv.FONT_HERSHEY_SIMPLEX
    area=cv.contourArea(max_ct)
    cv.putText(frame, 'Gesture : ', (260, 35), font, 1, (255, 0, 0), 1)
    if dist(top,cntr)>110 and dist(l,cntr)<90:
        cv.putText(frame, 'UP', (430, 35), font, 1 ,(255, 255, 0), 2)
    elif dist(top,cntr)<90 and dist(r,cntr)>100:
        cv.putText(frame, 'RIGHT', (430, 35), font, 1, (0, 255, 0), 2)
    elif dist(top,cntr)>110 and dist(l,cntr)>100:
        cv.putText(frame, 'LEFT', (430, 35), font, 1, (255, 0, 255), 2)
    elif dist(top,cntr)<110 and dist(l,cntr)<100 and dist(r,cntr)<90:
        cv.putText(frame, 'DOWN', (430, 35), font, 1, (0, 255, 255), 2)

    cv.imshow('res',res) 
    cv.imshow('img',frame)
    
    cv.circle(roi, l, 8, (0, 0, 255), -1)
    cv.circle(roi, r, 8, (0, 255, 0), -1)
    cv.circle(roi, top, 8, (255, 0, 0), -1)
    cv.circle(roi, cntr, 8, (0, 255, 255), -1)
    
    
    
    cv.imshow("roi",roi)
    if cv.waitKey(40) == 27: # Esc to quit the video windows
        break

cv.destroyAllWindows()
cap.release()   
