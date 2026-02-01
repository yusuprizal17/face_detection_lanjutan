import cv2
import numpy as np

face_ref = cv2.CascadeClassifier("face_ref.xml")
camera = cv2.VideoCapture(0)

model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.yml")
label_map = np.load("label_map.npy", allow_pickle=True).item()

def face_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(
        gray,
        scaleFactor=1.2,   # jangan terlalu kecil
        minNeighbors=7,    # SEMAKIN BESAR → makin selektif
        minSize=(80, 80)   # abaikan objek kecil
)
    return faces, gray

def shape(frame):
    faces, gray = face_detection(frame)

    for x, y, w, h in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = model.predict(face_img)

        name = label_map[label] if confidence < 80 else "Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = camera.read()
        shape(frame)
        cv2.imshow("Alburaong Detected Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()

if __name__ == '__main__':
    main()
