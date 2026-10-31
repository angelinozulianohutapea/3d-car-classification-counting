import argparse, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector import Detector

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/video.mp4")
ap.add_argument("--model", default="centernet-3d-bbox.pth")
ap.add_argument("--frame", type=int, default=0)
args = ap.parse_args()

cap = cv2.VideoCapture(args.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
ok, f = cap.read()
f = cv2.resize(f, (1280, 720))

det = Detector(args.model, conf=0.05)
hm, _, _ = det.infer(f)
print("skor puncak tertinggi di frame ini:", float(hm.max()))

big = cv2.resize(hm, (1280, 720), interpolation=cv2.INTER_CUBIC)
heat = cv2.applyColorMap((np.clip(big, 0, 1) * 255).astype("uint8"), cv2.COLORMAP_JET)
out = cv2.addWeighted(f, 0.5, heat, 0.5, 0)
cv2.imwrite("experiments/out/heatmap_check.jpg", out)
print("tersimpan: experiments/out/heatmap_check.jpg")
