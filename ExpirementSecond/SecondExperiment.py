import numpy as np
import cv2
import time
import datetime
import os


cap = cv2.VideoCapture("SecondShenariyLeft.mp4")
cap1 = cv2.VideoCapture("Secomdvideomiddle.mp4")

fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg1 = cv2.createBackgroundSubtractorMOG2()

# Fps count
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second camera: {0}".format(fps))

# Number of frames to capture
num_frames = 1

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while(1):
    # Start time
    start = time.time()

    ret, frame = cap.read()
    ret, frame1 = cap1.read()

    # Draw image size
    font = cv2.FONT_ITALIC
    text = '('+ str(cap.get(3)) + ' X ' + str(cap.get(4)) + ')'
    frame = cv2.putText(frame, text, (10, 25), font, 0.5, (231, 230, 229), 2, cv2.LINE_AA)
    frame1 = cv2.putText(frame1, text, (10, 25), font, 0.5, (231, 230, 229), 2, cv2.LINE_AA)

    # Draw real time and date
    font = cv2.FONT_ITALIC
    text = '('+ str(cap.get(3)) + ' X ' + str(cap.get(4)) + ')'
    datet = str(datetime.datetime.now())
    frame = cv2.putText(frame, datet, (10, 50), font, 0.5, (231, 230, 229), 2, cv2.LINE_AA)
    frame1 = cv2.putText(frame1, datet, (10, 50), font, 0.5, (231, 230, 229), 2, cv2.LINE_AA)

    # Draw a fps
    font = cv2.FONT_ITALIC
    frame = cv2.putText(frame, "FPS: " + str(round(fps)), (550, 25), font, 0.5, (231, 230, 229), 2, cv2.LINE_AA)
    frame1 = cv2.putText(frame1, "FPS: " + str(round(fps)), (550, 25), font, 0.5, (231, 230, 229), 2, cv2.LINE_AA)



    fgmask = fgbg.apply(frame)
    fgmask1 = fgbg1.apply(frame1)
    
    # Detect contours in the frame.
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours1, _ = cv2.findContours(fgmask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the frame to draw bounding boxes around the detected cars.
    frameCopy = frame.copy()
    frameCopy1 = frame1.copy()
    
    # loop over each contour found in the frame.
    for cnt in contours:
        
        # Make sure the contour area is somewhat higher than some threshold to make sure its a car and not some noise.
        if cv2.contourArea(cnt) > 4500:
            
            # Retrieve the bounding box coordinates from the contour.
            x, y, width, height = cv2.boundingRect(cnt)
            
            # Draw a bounding box around the car.
            cv2.rectangle(frameCopy, (x , y), (x + width, y + height),(0, 0, 255), 2)
                
            # Write Car Detected near the bounding box drawn.
            cv2.putText(frameCopy, 'Object detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
    for cnt1 in contours1:
        
        # Make sure the contour area is somewhat higher than some threshold to make sure its a car and not some noise.
        if cv2.contourArea(cnt1) > 4500:
            
            # Retrieve the bounding box coordinates from the contour.
            x, y, width, height = cv2.boundingRect(cnt1)
            
            

            # Draw a bounding box around the car.
            cv2.rectangle(frameCopy1, (x , y), (x + width, y + height),(0, 0, 255), 2)
                
            # Write Car Detected near the bounding box drawn.
            cv2.putText(frameCopy1, 'Objects detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
    # End time for whole program running 120 frames
    end = time.time()
 
    # Time elapsed
    seconds = end - start

    if seconds!= 0:
        # Calculate frames per second
        fps = num_frames / seconds

    cv2.imshow('RightCamera',frameCopy)
    cv2.imshow('MiddleCamera',frameCopy1)
    #cv2.imshow('fgmask MOG2',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()