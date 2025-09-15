import cv2 as cv
import numpy as np 

# Load Haar Cascade
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Labels
people = ["Elon", "Jeff Bezos", "Ronaldo", "Salah"]

# Load trained model
face_recognizer = cv.face_LBPHFaceRecognizer_create()
face_recognizer.read('face_train.yml')

# Load image
img = cv.imread(r'D:\SVU Rasing Team\Task 3 == Face Reco\-Face-Recognition-System\valua\22.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect faces
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in faces_rect:
    face_roi = gray[y:y+h, x:x+w]

    labels, confidence = face_recognizer.predict(face_roi)
    print(f'Label = {labels} with confidence = {confidence}')

    if confidence < 70:
        name = people[labels]
    else:
        name = "Unknown"

    cv.putText(img, f'{name} ({int(confidence)})', (x, y-10),
               cv.FONT_HERSHEY_DUPLEX, 1.0, (255,255,0), 2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
