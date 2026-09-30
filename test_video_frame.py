import cv2

video = "DLR_HT_060000_060100.mp4"

cap = cv2.VideoCapture(video)

if not cap.isOpened():
    print("Video tidak bisa dibuka")
    exit()

ret, frame = cap.read()

if not ret:
    print("Frame tidak berhasil dibaca")
    cap.release()
    exit()

print("Ukuran frame:", frame.shape)

cv2.imwrite("dlr_frame.jpg", frame)

cap.release()

print("Frame berhasil disimpan: dlr_frame.jpg")
