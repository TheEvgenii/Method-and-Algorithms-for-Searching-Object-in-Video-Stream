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
cap = cv2.VideoCapture(0)


# Fps count
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second camera: {0}".format(fps))

# Number of frames to capture
num_frames = 1
  
# change resolution

def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

make_480p()  # set width as 640 and set height as 480

# Getting values for width and height
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Using Haar Classifiers for face and body detection with CUDA
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

recodring = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

# Building frame size
frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")    # Format of the video

# Setting wallet
while True:

    # Start time
    start = time.time()

    _, frame = cap.read()

    if frame is None:
        print("No frame")
        break

    # New image is grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw image size
    font = cv2.FONT_ITALIC
    text = '('+ str(cap.get(3)) + ' X ' + str(cap.get(4)) + ')'
    frame = cv2.putText(frame, text, (10, 50), font, 1, (231, 230, 229), 2, cv2.LINE_AA)

    # Draw real time and date
    font = cv2.FONT_ITALIC
    text = '('+ str(cap.get(3)) + ' X ' + str(cap.get(4)) + ')'
    datet = str(datetime.datetime.now())
    frame = cv2.putText(frame, datet, (10, 100), font, 1, (231, 230, 229), 2, cv2.LINE_AA)

    # Draw a fps
    font = cv2.FONT_ITALIC
    frame = cv2.putText(frame, "FPS: " + str(round(fps)), (500, 50), font, 1, (231, 230, 229), 2, cv2.LINE_AA)

    # List of all of the faces positions that exist
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # List of all of the bodies positions that exist
    bodies = body_cascade.detectMultiScale(gray, 1.1, 1)

    if len(faces) + len(bodies) > 0:
        make_720p() # Change resolution to a hire quallity
        # Taking frame new frame size
        frame_size = (int(cap.get(3)), int(cap.get(4)))
        if recodring:
            timer_started = False
        else:
            # New recording
            recodring = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(
                f"{current_time}.mp4", fourcc, 20, frame_size)
            captureImage()
            sendemail()                            # Sends an email after detecting an object
            print("Started Recording!")
            
    elif recodring:
        if timer_started:
            # If current time passed value after detection
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                recodring = False
                timer_started = False
                out.release()
                print('Stop Recording!')
                make_480p() # Change resolution to a lower quallity
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if recodring:
        out.write(frame)
    
    # Draw rectangle on the screen for faces
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x,y), (x + width, y + height), (255, 0, 0), 3)

    # Draw rectangle on the screen for bodies
    for (x, y, width, height) in bodies:
        cv2.rectangle(frame, (x,y), (x + width, y + height), (255, 0, 0), 2)

    # End time for whole program running 120 frames
    end = time.time()

    # Time elapsed
    seconds = end - start
    
    # Calculate frames per second
    fps = num_frames / seconds

    cv2.imshow("Camera", frame)

    #Stop program
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()