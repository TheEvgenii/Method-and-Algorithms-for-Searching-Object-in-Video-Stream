import cv2
import numpy as np

# load a video
cap = cv2.VideoCapture('carsvid.mp4')

# You can set custom kernel size if you want.
kernel = None

# you can optionally work on the live web cam
#cap = cv2.VideoCapture(0)

# create the background object, you can choose to detect shadows or not (if True they will be shown as gray)
backgroundobject = cv2.createBackgroundSubtractorMOG2( history = 2, detectShadows = True )

while(1):
    ret, frame = cap.read()  
    if not ret:
        break
        
    # apply the background object on each frame
    fgmask = backgroundobject.apply(frame)

    # also extracting the real detected foreground part of the image (optional)
    real_part = cv2.bitwise_and(frame,frame,mask=fgmask)
    
    # making fgmask 3 channeled so it can be stacked with others
    fgmask_3 = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    
    # Stack all three frames and show the image
    stacked = np.hstack((fgmask_3,frame,real_part))
    cv2.imshow('All three',cv2.resize(stacked,None,fx=0.65,fy=0.65))
 
    k = cv2.waitKey(30) &  0xff
    if k == 27:
        break
   
cap.release()
cv2.destroyAllWindows()
