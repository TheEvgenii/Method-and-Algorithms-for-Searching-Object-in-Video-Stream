import cv2
import numpy as np

img = cv2.imread("photo.jpeg")

template = cv2.imread("airpods.png")

cv2.imshow("template", img)
cv2.waitKey(0)
cv2.destroyAllWindows()