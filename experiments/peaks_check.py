import argparse, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detector as D
from detector import Detector

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/video.mp4")
ap.add_argument("--model", default="centernet-3d-bbox.pth")
ap.add_argument("--frame", type=int, default=120)
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
ok, f = cap.read()
f = cv2.resize(f, (1280, 720))

det = Detector(a.model, conf=0.01)
hm, off, _ = det.infer(f)
pts = sorted(D.decode_points(hm, off, 0.01), key=lambda p: -p["score"])
for n, p in enumerate(pts):
    print(f'#{n:2d} x={p["x"]:6.0f} y={p["y"]:6.0f} skor={p["score"]:.3f}')
    c = (0, 255, 0) if p["score"] >= 0.15 else (0, 0, 255)
    cv2.circle(f, (int(p["x"]), int(p["y"])), 6, c, -1)
    cv2.putText(f, f'#{n} {p["score"]:.2f}', (int(p["x"]) + 8, int(p["y"])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
cv2.imwrite("experiments/out/peaks_check.jpg", f)
print("tersimpan: experiments/out/peaks_check.jpg")
