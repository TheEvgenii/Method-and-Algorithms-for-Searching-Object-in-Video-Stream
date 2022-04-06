#  MOG.py
#  Evgenii_Litvinov
#  COSC4399
#  Code was written in Python language
#  Created by Evgenii Litvinov on 11/25/21.
#  Reference: https://bleedai.com/jupyter_notebook/Vehicle_Detection.html
#

import numpy as np
import cv2

cap = cv2.VideoCapture("FirstCamera1.mp4")

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    
  	# Detect contours in the frame.
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the frame to draw bounding boxes around the detected objects.
    frameCopy = frame.copy()
    
    # Loop over each contour found in the frame.
    for cnt in contours:
        
        # Make sure the contour area is somewhat higher than some threshold to make sure its a movment and not some noise.
        if cv2.contourArea(cnt) > 250:
            
            # Retrieve the bounding box coordinates from the contour.
            x, y, width, height = cv2.boundingRect(cnt)
            
            # Draw a bounding box around the object.
            cv2.rectangle(frameCopy, (x , y), (x + width, y + height),(0, 0, 255), 2)
            
            # Write Object Detected near the bounding box drawn.
            cv2.putText(frameCopy, 'Object detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)

    cv2.imshow('frameCopy MOG',frameCopy)
    cv2.imshow('fgmask MOG',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()