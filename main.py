import cv2


face_ref = cv2.CascadeClassifier("face_ref.xml")
camera = cv2.VideoCapture(0)

#Deteksi wajah
def face_detection(frame):

    #Mengganti warna menjadi hitam putih (Agar sytem deteksinya ringan)
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1)
    return faces

#Membuat kotak didalam wajah
def shape(frame):
     #x = kiri-kanan , y = atas bawah, w = lebar, h = Tinggi
    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

#Keluar jendela windows
def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

#Menajalankan kamera
def main():
    while True:
        _, frame = camera.read()
        shape(frame)
        cv2.imshow("Alburaong Detected Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()
    
if __name__ == '__main__':
    main()