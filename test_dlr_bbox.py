import cv2
from object_detector import ObjectDetector

image = cv2.imread("dlr_frame.jpg")

if image is None:
    print("Frame tidak ditemukan")
    exit()

detector = ObjectDetector(
    "centernet-3d-bbox.pth",
    0.2
)

boxes = detector.detect(image)

print("Jumlah deteksi:", len(boxes))

for i, box in enumerate(boxes):
    print(i + 1, box)

    x = int(box["x"])
    y = int(box["y"])

    cv2.circle(image, (x, y), 8, (0, 0, 255), -1)

    cv2.putText(
        image,
        f"Vehicle {i + 1}",
        (x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

cv2.imwrite("dlr_detection.jpg", image)

print("Hasil disimpan: dlr_detection.jpg")
