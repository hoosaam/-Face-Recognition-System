import cv2 as cv
import numpy as np 


haar_cascade=cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
people = ["Elon", "Jeff Bezos", "ronaldo","salah"]

face_recognizer = cv.face.LBPHFaceRecognizer.create()
face_recognizer.read('face_train.yml')

img=cv.imread(r'D:\SVU Rasing Team\Task 3 == Face Reco\valua\2.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

for (x,y,w,h) in face_rect:
     face_roi=gray[y:y+h,x:x+w]
     labels,confidence=face_recognizer.predict(face_roi)
     print(f'Label = {labels} with a confidence = {confidence}')
     print(w)
     cv.putText(img, people[labels],(x,y-10),cv.FONT_HERSHEY_DUPLEX,1.0,(255,255,0),thickness=2)
     cv.rectangle(img , (x,y),(x+w,y+h),(0,255,0),thickness=1)


cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()



