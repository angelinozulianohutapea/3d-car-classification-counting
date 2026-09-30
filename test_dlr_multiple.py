import cv2
import os
from object_detector import ObjectDetector

video = "DLR_HT_060000_060100.mp4"

cap = cv2.VideoCapture(video)

if not cap.isOpened():
    print("Video tidak bisa dibuka")
    exit()

detector = ObjectDetector(
    "centernet-3d-bbox.pth",
    0.2
)

os.makedirs("dlr_test_frames", exist_ok=True)

# Ambil frame setiap 200 frame
frame_numbers = [0, 200, 400, 600, 800, 1000]

for frame_number in frame_numbers:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if not ret:
        print("Frame", frame_number, "tidak bisa dibaca")
        continue

    # Resize sesuai ukuran input model
    frame = cv2.resize(frame, (1280, 720))

    boxes = detector.detect(frame)

    print("--------------------------------")
    print("Frame:", frame_number)
    print("Jumlah deteksi:", len(boxes))

    for i, box in enumerate(boxes):

        x = int(box["x"])
        y = int(box["y"])

        print(
            i + 1,
            "x =", x,
            "y =", y
        )

        cv2.circle(
            frame,
            (x, y),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"Vehicle {i + 1}",
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    filename = f"dlr_test_frames/frame_{frame_number}.jpg"

    cv2.imwrite(filename, frame)

    print("Disimpan:", filename)

cap.release()

print("--------------------------------")
print("SELESAI")
