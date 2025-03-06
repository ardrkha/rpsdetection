import cv2
from ultralytics import YOLO

# Load model YOLO
model_path = "best.pt"  # Ganti dengan path model yang sesuai
model = YOLO(model_path)

# Buka webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame agar tidak mirror
    frame = cv2.flip(frame, 1)

    # Ubah ukuran gambar sebelum inferensi agar lebih ringan
    results = model.predict(frame, imgsz=320, verbose=False)

    # Visualisasi hasil deteksi langsung pada frame
    annotated_frame = results[0].plot()

    # Tampilkan hasil
    cv2.imshow("Rock-Paper-Scissors Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Tutup kamera dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
