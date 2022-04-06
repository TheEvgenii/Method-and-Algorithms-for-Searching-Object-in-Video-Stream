#
#  FirstExerimentStudentProctor.py
#  Evgenii_Litvinov
#  COSC4399
#  Code was written in Python language
#  Created by Evgenii Litvinov on 01/25/22.
#


import numpy as np
import cv2
import time
import datetime
import os
import smtplib    # Sends email
import imghdr
from email.message import EmailMessage


# Sends an email 
def sendemail():
    msg = EmailMessage()
    msg['Subject'] = 'Object was detected!'
    msg['From'] = EMAIL_USER
    msg['To'] = 'alanmatt2000@gmail.com'
    msg.set_content('Object has been recorded. Image attached: ')

    # Send a capture of the image from ditected object
    with open('Object_detected_0.png', 'rb') as f:
        file_data = f.read()
        file_type = imghdr.what(f.name)
        file_name = f.name

    msg.add_attachment(file_data, maintype='image', subtype=file_type,filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)

def captureImage(IMG_COUNTER = 0):
    img_name = "Object_detected_{}.png".format(IMG_COUNTER)
    cv2.imwrite(img_name,frame)
    print("Screenshot taken")
    IMG_COUNTER+=1

# Get email addres and password from invirement vairablee
EMAIL_USER = os.environ.get('EMAIL_ADDRESS')
EMAIL_PASS = os.environ.get('EMAIL_PASSWORD')


# Access webcam 
cap = cv2.VideoCapture("LeftCamera.mp4")
cap1 = cv2.VideoCapture("RightCamera.mp4")

if cap is None:
    print("No first File")

if cap1 is None:
    print("No second File")


fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg1 = cv2.createBackgroundSubtractorMOG2()

# Fps count
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second camera: {0}".format(fps))

# Number of frames to capture
num_frames = 1

recodringLeft = False
recodringRight = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 10

counting = 0
counting2 = 0
screenshot = 1

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")    # Format of the video

# Setting wallet
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
        
        # Make sure the contour area is somewhat higher than some threshold to make sure its a object and not some noise.
        if cv2.contourArea(cnt) > 500:
            
            # Retrieve the bounding box coordinates from the contour.
            x, y, width, height = cv2.boundingRect(cnt)
    
            if x> 570 and y > 150 and y!=0:
                counting +=1

                if counting>500:
                    # Taking frame new frame size
                    frame_size = (int(cap.get(3)), int(cap.get(4)))


                    if recodringLeft:
                        timer_started = False


                    else:
                        # New recording
                        recodringLeft = True
                        recodringRight = True
                        current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                        out = cv2.VideoWriter(
                            f"LeftCamera_{current_time}.mp4", fourcc, 20, frame_size)

                        current_time2 = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                        outRight = cv2.VideoWriter(
                            f"RightCamera_{current_time2}.mp4", fourcc, 20, frame_size)

                        if screenshot <= 2 and x> 570 and y > 160:
                            captureImage()
                            screenshot += 1
                            sendemail()                            # Sends an email after detecting an object
                        print("Started Recording Left Camera!")
                        print("Started Recording Right Camera!")



                    # Draw a bounding box around the object.
                    cv2.rectangle(frameCopy, (x , y), (x + width, y + height),(0, 0, 255), 2)
                    
                    # Write object Detected near the bounding box drawn.
                    cv2.putText(frameCopy, 'Object detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
            

            elif recodringLeft:
                if timer_started:
                    # If current time passed value after detection
                    if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                        recodringLeft = False
                        recodringRight = False
                        timer_started = False
                        out.release()
                        outRight.release()
                        print('Stop Recording Left Camera!')
                        print('Stop Recording Right Camera!')
                        counting = 0
                        counting2 = 0
                       
                else:
                    timer_started = True
                    detection_stopped_time = time.time()

            if recodringLeft:
                out.write(frame)
                outRight.write(frame1)



    for cnt1 in contours1:
        
        # Make sure the contour area is somewhat higher than some threshold to make sure its a object and not some noise.
        if cv2.contourArea(cnt1) > 500:
            
            # Retrieve the bounding box coordinates from the contour.
            x, y, width, height = cv2.boundingRect(cnt1)

            
            if x < 70 and y > 340 and y!=0:
                counting2 +=1
                print(counting2,counting)
                if counting2>500:


                    # Taking frame new frame size
                    frame_size = (int(cap.get(3)), int(cap.get(4)))


                    if recodringRight:
                        timer_started = False


                    else:
                        # New recording
                        recodringLeft = True
                        recodringRight = True
                        current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                        out = cv2.VideoWriter(
                            f"LeftCamera_{current_time}.mp4", fourcc, 20, frame_size)

                        current_time2 = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                        outRight = cv2.VideoWriter(
                            f"RightCamera_{current_time2}.mp4", fourcc, 20, frame_size)

                        if screenshot == 1:
                            captureImage()
                            screenshot += 1
                            sendemail()                            # Sends an email after detecting an object
                        print("Started Recording Left Camera!")
                        print("Started Recording Right Camera!")
                        counting = 0
                        counting2 = 0



                    # Draw a bounding box around the object.
                    cv2.rectangle(frameCopy1, (x , y), (x + width, y + height),(0, 0, 255), 2)
                    
                    # Write Car Detected near the bounding box drawn.
                    cv2.putText(frameCopy1, 'Object detected25', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)

            elif recodringLeft:
                if timer_started:
                    # If current time passed value after detection
                    if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                        recodringLeft = False
                        recodringRight = False
                        timer_started = False
                        out.release()
                        outRight.release()
                        print('Stop Recording Left Camera!')
                        print('Stop Recording Right Camera!')
                       
                else:
                    timer_started = True
                    detection_stopped_time = time.time()

            if recodringLeft:
                out.write(frame)
                outRight.write(frame1)


    # End time for whole program running 120 frames
    end = time.time()
 
    # Time elapsed
    seconds = end - start

    if seconds!= 0:
        # Calculate frames per second
        fps = num_frames / seconds

    # making fgmask 3d so it can be stacked with others
    fgmask_2 = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)    

    # Stack the original framen and extracted foreground. 
    stacked = np.hstack((frameCopy, frameCopy1,fgmask_2))       
    cv2.imshow('LeftCamera & RightCamera & fgmask', cv2.resize(stacked, None, fx=1, fy=1))

    #cv2.imshow('LeftCamera',frameCopy)
    #cv2.imshow('RightCamera',frameCopy1)
    #cv2.imshow('fgmask MOG2',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()