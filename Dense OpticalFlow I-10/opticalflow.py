
#  #Dense
#  OpticalFlow.py
#  Evgenii_Litvinov
#  COSC4399
#  Code was written in Python language
#  Created by Evgenii Litvinov on 10/20/21.
#  Reference: https://gist.github.com/RodolfoFerro/11d39fad57e21b5e85fe4d4a906cf098
#

import numpy as np
import cv2
import time
import datetime

# Global vars:
STEP = 16
QUIVER = (0, 255, 0)

def draw_flow(img, flow, step=STEP):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, QUIVER)

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr



cap = cv2.VideoCapture("I-10RoadHouston.mp4")

ret, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


while True:

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5),0)

    # start time to calculate FPS
    start = time.time()

    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray


    # End timee
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)


    img_3 = np.concatenate((img,draw_flow(gray, flow)), axis=1)

    # Draw a fps
    font = cv2.FONT_ITALIC
    img_3 = cv2.putText(img_3, "FPS: " + str(round(fps)), (550, 25), font, 0.5, (231, 230, 229), 2, cv2.LINE_AA)

    # Draw real time and date
    font = cv2.FONT_ITALIC
    text = '('+ str(cap.get(3)) + ' X ' + str(cap.get(4)) + ')'
    datet = str(datetime.datetime.now())
    img_3 = cv2.putText(img_3, datet, (10, 50), font, 0.5, (231, 230, 229), 2, cv2.LINE_AA)

    # Draw image size
    font = cv2.FONT_ITALIC
    text = '('+ str(cap.get(3)) + ' X ' + str(cap.get(4)) + ')'
    img_3 = cv2.putText(img_3, text, (10, 25), font, 0.5, (231, 230, 229), 2, cv2.LINE_AA)
    
    cv2.imshow('Dense Optical Flow', img_3)


    key = cv2.waitKey(30)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()