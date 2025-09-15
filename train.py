import cv2 as cv
import numpy as np
import os

# Load the model

people = ["Elon", "Jeff Bezos", "ronaldo","salah"]

Dir= r'D:\SVU Rasing Team\Task 3 == Face Reco\-Face-Recognition-System\Persons'

haar_cascade=cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(Dir, person)
        label = people.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            if img_array is None:
                print(f"Could not load image: {img_path}")
                continue
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            
            if len(faces_rect) == 0:
                print(f"No faces detected in: {img_path}")
                continue
                
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
    return features, labels


create_train()
print(len(features))
print(len(labels))
print("_____________________________(:-----------:)___________________________")

features =np.array(features,dtype='object')
labels=np.array(labels)
face_recognizer = cv.face_LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)
face_recognizer.save('face_train.yml')
np.save('feature.npy',features)
np.save('labels.npy',labels)

print("_____________________________(:-----------:)___________________________")







