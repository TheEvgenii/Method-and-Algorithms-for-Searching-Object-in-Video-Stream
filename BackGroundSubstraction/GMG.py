import numpy as np
import cv2

cap = cv2.VideoCapture("FirstCamera1.mp4")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
  	# Detect contours in the frame.
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the frame to draw bounding boxes around the detected cars.
    frameCopy = frame.copy()
    
    # loop over each contour found in the frame.
    for cnt in contours:
        
        # Make sure the contour area is somewhat higher than some threshold to make sure its a car and not some noise.
        if cv2.contourArea(cnt) > 2000:
            
            # Retrieve the bounding box coordinates from the contour.
            x, y, width, height = cv2.boundingRect(cnt)
            
            # Draw a bounding box around the car.
            cv2.rectangle(frameCopy, (x , y), (x + width, y + height),(0, 0, 255), 2)
            
            # Write Car Detected near the bounding box drawn.
            cv2.putText(frameCopy, 'Object detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)

    cv2.imshow('frameCopy GMG',frameCopy)
    cv2.imshow('fgmask GMG',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()