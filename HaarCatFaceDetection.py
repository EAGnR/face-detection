import numpy as np
import cv2

SF = 1.05 # Scale Factor
MN = 3 # Min Neighbors

cat_face_cascade = cv2.CascadeClassifier('OpenCV/haarcascades/haarcascade_frontalcatface.xml')
cat_face_cascade_ext = cv2.CascadeClassifier('OpenCV/haarcascades/haarcascade_frontalcatface_extended.xml')

img = cv2.imread('images/dog-and-cat.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = list(cat_face_cascade.detectMultiScale(gray, scaleFactor = SF, minNeighbors = MN))
faces_ext = list(cat_face_cascade_ext.detectMultiScale(gray, scaleFactor = SF, minNeighbors = MN))

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

for (x,y,w,h) in faces_ext:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindows()