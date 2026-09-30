import cv2
from object_detector import ObjectDetector
from multi_object_detection import showbox

image = cv2.imread("carla_dataset/images/image_0001.png")

if image is None:
    print("Gambar tidak ditemukan")
    exit()

print("Ukuran gambar:", image.shape)

detector = ObjectDetector(
    "centernet-3d-bbox.pth",
    0.2
)

boxes = detector.detect(image)

print("Jumlah deteksi:", len(boxes))

result = showbox(image, boxes)

cv2.imwrite(
    "carla_3d_bbox.jpg",
    result
)

print("Hasil disimpan: carla_3d_bbox.jpg")
