import cv2
import os

# Nama orang yang akan dicapture
person_name = input("Masukkan nama orang: ").strip()

# Folder dataset
dataset_path = "dataset"
person_path = os.path.join(dataset_path, person_name)

# Buat folder jika belum ada
os.makedirs(person_path, exist_ok=True)

# Load Haarcascade
face_ref = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Buka kamera
camera = cv2.VideoCapture(0)

count = 0
print("Tekan 's' untuk capture wajah, 'q' untuk keluar.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Gagal membuka kamera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Tampilkan kotak di wajah
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # tekan 's' untuk simpan wajah
        if len(faces) == 0:
            print("Tidak ada wajah terdeteksi, coba lagi...")
            continue
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            count += 1
            file_path = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)
            print(f"Wajah tersimpan: {file_path}")
    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print(f"Capture selesai, total wajah tersimpan: {count}")
