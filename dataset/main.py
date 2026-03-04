import cv2
import numpy as np
import os
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

dataset_path = "dataset"
known_embeddings = []
known_names = []

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(img_rgb)
        if faces:
            x, y, w, h = faces[0]['box']
            face = img_rgb[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))

            embedding = embedder.embeddings([face])[0]
            known_embeddings.append(embedding)
            known_names.append(person)

print("Dataset loaded successfully!")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    for face_data in faces:
        x, y, w, h = face_data['box']
        face = img_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))

        embedding = embedder.embeddings([face])[0]

        distances = [np.linalg.norm(embedding - known_embedding)
                     for known_embedding in known_embeddings]

        if distances:
            min_distance = min(distances)
            index = distances.index(min_distance)

            if min_distance < 0.8:
                name = known_names[index]
            else:
                name = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()