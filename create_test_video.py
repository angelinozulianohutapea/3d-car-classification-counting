import cv2

image = cv2.imread("periph.jpg")

if image is None:
    print("Gambar tidak ditemukan")
    exit()

height, width = image.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(
    "test_video.mp4",
    fourcc,
    10,
    (width, height)
)

for _ in range(100):
    video.write(image)

video.release()

print("Video berhasil dibuat: test_video.mp4")
