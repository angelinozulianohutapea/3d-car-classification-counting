import cv2
from object_detector import ObjectDetector

image = cv2.imread("dlr_frame.jpg")

if image is None:
    print("Frame tidak ditemukan")
    exit()

print("Ukuran asli:", image.shape)

# Resize ke ukuran input model
image = cv2.resize(image, (1280, 720))

print("Ukuran setelah resize:", image.shape)

detector = ObjectDetector(
    "centernet-3d-bbox.pth",
    0.2
)

boxes = detector.detect(image)

print("==============================")
print("HASIL DETEKSI SETELAH RESIZE")
print("==============================")
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

cv2.imwrite("dlr_detection_resize.jpg", image)

print("Hasil disimpan: dlr_detection_resize.jpg")
