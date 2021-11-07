import cv2
import time
import datetime

#Access webcam 
cap = cv2.VideoCapture(0)

# Getting values for width and height
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    _, frame = cap.read()

    # New image is grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # List of all of the faces positions that exist
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # List of all of the bodies positions that exist
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(bodies) > 0:
        if recodring:
            timer_started = False
        else:
            # New recording
            recodring = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(
                f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Started Recording!")
    elif recodring:
        if timer_started:
            # If current time passed value after detection
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                recodring = False
                timer_started = False
                out.release()
                print('Stop Recording!')
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
        cv2.rectangle(frame, (x,y), (x + width, y + height), (255, 0, 0), 3)

    # Draw image size
    font = cv2.FONT_ITALIC
    text = '('+ str(cap.get(3)) + ' X ' + str(cap.get(4)) + ')'
    frame = cv2.putText(frame, text, (10, 50), font, 1, (231, 230, 229), 2, cv2.LINE_AA)

    # Draw real time and date
    font = cv2.FONT_ITALIC
    text = '('+ str(cap.get(3)) + ' X ' + str(cap.get(4)) + ')'
    datet = str(datetime.datetime.now())
    frame = cv2.putText(frame, datet, (10, 100), font, 1, (231, 230, 229), 2, cv2.LINE_AA)



    cv2.imshow("Camera", frame)

    #Stop program
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()