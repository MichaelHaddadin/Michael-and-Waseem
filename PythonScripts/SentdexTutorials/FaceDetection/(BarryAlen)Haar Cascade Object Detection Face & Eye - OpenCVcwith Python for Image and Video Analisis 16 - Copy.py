import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('MoreEyeCascades7000.xml')
#http://alereimondo.no-ip.org/OpenCV/34
mouth_cascade = cv2.CascadeClassifier('Mouth_Cascade.xml')
#http://alereimondo.no-ip.org/OpenCV/34
nose_cascade = cv2.CascadeClassifier('Nose_Cascade.xml')

cap = cv2.VideoCapture("Godspeed vs Flash Fight Scene The Flash 6x18 [HD].mp4")

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
           cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #mouth = mouth_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in mouth:
           #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,100,125),2)
        #nose = nose_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in nose:
           #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(125,100,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
