import cv2
import os
import numpy as np

face_ref = cv2.CascadeClassifier("face_ref.xml")
dataset_path = "dataset"

faces = []
labels = []
label_map = {}
current_label = 0

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = face_ref.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in detected_faces:
            faces.append(gray[y:y+h, x:x+w])
            labels.append(current_label)

    current_label += 1

faces = np.array(faces, dtype=object)
labels = np.array(labels)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)

model.save("face_model.yml")
np.save("label_map.npy", label_map)

print("Training selesai!")
