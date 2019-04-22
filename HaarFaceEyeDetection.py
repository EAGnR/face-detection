import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('OpenCV/haarcascades/haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier('OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
face_cascade_alt2 = cv2.CascadeClassifier('OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')
face_cascade_alt_tree = cv2.CascadeClassifier('OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml')
eye_cascade = cv2.CascadeClassifier('OpenCV/haarcascades/haarcascade_eye.xml')
eyeglasses_cascade = cv2.CascadeClassifier('OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

img = cv2.imread('images/faces-in-crowd.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = list(face_cascade_alt.detectMultiScale(gray, 1.04, 3))
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = list(eye_cascade.detectMultiScale(roi_gray, minNeighbors = 5))
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,0,255), 2)

cv2.imshow('img',img)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    vid_faces = list(face_cascade_alt.detectMultiScale(gray_frame, 1.3, 5))
    for (x,y,w,h) in vid_faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = list(eye_cascade.detectMultiScale(roi_gray))
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,0,255), 2)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(5) & 0xFF

    if k == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()