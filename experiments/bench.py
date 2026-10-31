import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, torch
from detector import Detector

cap = cv2.VideoCapture("data/video.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 285)
ok, f = cap.read()
f = cv2.resize(f, (1280, 720))
det = Detector("centernet-3d-bbox.pth", scales=(1.0,))
print("core logis:", os.cpu_count())

for th in sorted({1, 2, 4, max(1, (os.cpu_count() or 4) // 2), os.cpu_count() or 4}):
    torch.set_num_threads(th)
    for sc in (1.0, 1.25, 1.5):
        det.infer(f, sc)  # pemanasan
        t = time.time()
        for _ in range(3):
            det.infer(f, sc)
        print(f"thread={th:2d} skala={sc}: {(time.time() - t) / 3 * 1000:6.0f} ms/frame")
