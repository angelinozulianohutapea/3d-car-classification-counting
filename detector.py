"""
Deteksi kendaraan dengan CenterNet versi sederhana (ResNet-18, stride 32).

Keluaran model per frame (peta ~23x40 untuk input 1280x720):
  heatmap [1,H,W], offset [2,H,W], ukuran w/h/l [3,H,W].
Puncak heatmap diambil dengan max-pooling 3x3 sebagai pengganti NMS.

Rujukan: X. Zhou, D. Wang, P. Krahenbuhl, "Objects as Points",
arXiv:1904.07850, 2019, bagian 4 (peak extraction). Arsitektur dan bobot
awal: proyek antoinekeller/3d_bbox_highway [kode asal].
Head heatmap hanya 1 kanal, jadi detektor TIDAK membedakan jenis kendaraan.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import transforms

import config as cfg


class CenterNet(nn.Module):
    """ResNet-18 sampai layer4 (512 kanal, stride 32) + 3 head."""

    def __init__(self):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        self.base_model = nn.Sequential(*list(base.children())[:-2])  # buang avgpool dan fc

        def head(out_ch, num_ch=512, head_conv=64):
            return nn.Sequential(
                nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, out_ch, kernel_size=1, stride=1),
            )

        self.outc = head(1)  # heatmap pusat kendaraan
        self.outo = head(2)  # offset sub-sel
        self.outr = head(3)  # ukuran w, h, l (rasio terhadap jarak)

    def forward(self, x):
        x = self.base_model(x)
        return self.outc(x), self.outo(x), self.outr(x)


def load_model(path, device="cpu"):
    model = CenterNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(device)
    model.eval()
    return model


def build_preprocess():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def select_pool(hm):
    """
    Pertahankan hanya sel yang maksimum lokal dalam jendela 3x3 (stride 1).
    Tidak bergantung pada ambang; ambang diterapkan di extract_peaks.
    Dengan stride 32, radius 1 sel ~ 32 piksel, sehingga dua mobil berdekatan
    tidak saling menekan.
    """
    t = torch.from_numpy(np.ascontiguousarray(hm)).unsqueeze(0).unsqueeze(0)
    keep = F.max_pool2d(t, kernel_size=3, stride=1, padding=1) == t
    return hm * keep.squeeze(0).squeeze(0).numpy()


def decode_points(hm, offset, conf):
    """
    hm: [H,W] probabilitas (sudah sigmoid), offset: [2,H,W].
    Mengembalikan puncak > conf sebagai dict {x, y (piksel), score, cy, cx}.
    Urutan baris-per-baris (row-major).
    """
    peaks = select_pool(hm)
    out = []
    for cy, cx in np.argwhere(peaks > conf):
        out.append(
            {
                "x": float((cx + offset[0, cy, cx]) * cfg.MODEL_SCALE),
                "y": float((cy + offset[1, cy, cx]) * cfg.MODEL_SCALE),
                "score": float(peaks[cy, cx]),
                "cy": int(cy),
                "cx": int(cx),
            }
        )
    return out


def decode_boxes(hm, offset, whl, conf):
    """Seperti decode_points, ditambah ukuran w, h, l; ukuran negatif dibuang."""
    boxes = []
    for p in decode_points(hm, offset, conf):
        w, h, l = (float(v) for v in whl[:, p["cy"], p["cx"]])
        if w < 0 or h < 0 or l < 0:
            continue
        boxes.append({"x": p["x"], "y": p["y"], "w": w, "h": h, "l": l, "score": p["score"]})
    return boxes


class Detector:
    def __init__(self, model_path, conf):
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.model = load_model(model_path, self.device)
        self.preprocess = build_preprocess()

    def infer(self, frame):
        """Frame BGR 1280x720 -> (heatmap [H,W] sigmoid, offset [2,H,W], whl [3,H,W])."""
        x = self.preprocess(frame).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            hm, offset, whl = self.model(x)
        hm = torch.sigmoid(hm).cpu().numpy()[0, 0]
        return hm, offset.cpu().numpy()[0], whl.cpu().numpy()[0]

    def detect(self, frame):
        hm, offset, whl = self.infer(frame)
        return decode_boxes(hm, offset, whl, self.conf)
