import cv2
import os

input_dir = "video_tracking_carla"
output_file = "tracking_result.mp4"

fps = 10
width = 1280
height = 720

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_file,
    fourcc,
    fps,
    (width, height)
)

for i in range(100):
    filename = os.path.join(input_dir, f"image_{i:04d}.png")
    frame = cv2.imread(filename)

    if frame is None:
        print(f"Frame tidak ditemukan: {filename}")
        continue

    frame = cv2.resize(frame, (width, height))
    out.write(frame)

    print(f"Frame {i + 1}/100")

out.release()

print("\nVideo berhasil dibuat:")
print(output_file)
