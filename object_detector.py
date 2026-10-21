import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from torchvision import transforms
import numpy as np


def select_legacy(hm, threshold):
    """
    Versi lama: tekan puncak yang berjarak <= 2 sel (stride 32 -> ~64 piksel).
    Dipertahankan agar hasil lama bisa direproduksi (--nms legacy).
    """
    pred = hm > threshold
    pred_centers = np.argwhere(pred)

    for i, ci in enumerate(pred_centers):
        for j in range(i + 1, len(pred_centers)):
            cj = pred_centers[j]
            if np.linalg.norm(ci - cj) <= 2:
                score_i = hm[ci[0], ci[1]]
                score_j = hm[cj[0], cj[1]]
                if score_i > score_j:
                    hm[cj[0], cj[1]] = 0
                else:
                    hm[ci[0], ci[1]] = 0

    return hm


def select_pool(hm):
    """
    Ambil puncak lokal dengan max-pooling 3x3 (stride 1), lalu buang sel yang
    bukan maksimum di lingkungannya.

    Dasar: Zhou, Wang, Krahenbuhl (2019), "Objects as Points", arXiv:1904.07850,
    bagian 4 (peak extraction): puncak diambil dengan max-pooling 3x3, sebagai
    pengganti NMS. Dengan stride 32, radius 1 sel ~ 32 piksel, jadi dua mobil
    yang berdekatan tidak ikut saling menekan seperti pada select_legacy.
    Tidak bergantung pada threshold: threshold diterapkan di pred2box.
    """
    t = torch.from_numpy(np.ascontiguousarray(hm)).unsqueeze(0).unsqueeze(0)
    keep = F.max_pool2d(t, kernel_size=3, stride=1, padding=1) == t
    return hm * keep.squeeze(0).squeeze(0).numpy()


def select(hm, threshold, mode="pool"):
    if mode == "legacy":
        return select_legacy(hm, threshold)
    if mode == "pool":
        return select_pool(hm)
    raise ValueError(f"mode NMS tidak dikenal: {mode}")


class ObjectDetector:
    """
    Initialize ObjectDetector by loading model
    Define confidence threshold
    Then preprocess, infer, post-process
    """

    INPUT_WIDTH = 1280
    INPUT_HEIGHT = 720
    MODEL_SCALE = 32

    def __init__(self, model_pth, conf_threhsold, nms="pool"):
        self.conf = conf_threhsold
        self.nms = nms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Device: {self.device} | NMS: {self.nms}")

        # Define and load model
        self.model = centernet()
        self.model.load_state_dict(
            torch.load(model_pth, map_location=torch.device("cpu"))
        )
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def detect(self, frame):
        """
        Detect objects on frame and return bounding boxes.
        Pre-process for model input
        Extract heatmaps, and compute predicted bounding boxes
        Setiap box berisi juga "score" (nilai heatmap di puncaknya).
        """

        input_tensor = self.preprocess(frame)

        # Inference (tanpa gradient: lebih hemat memori dan sedikit lebih cepat)
        with torch.no_grad():
            hm, offset, whl = self.model(
                input_tensor.to(self.device).float().unsqueeze(0)
            )

        hm = torch.sigmoid(hm)

        hm = hm.cpu().detach().numpy().squeeze(0).squeeze(0)
        offset = offset.cpu().detach().numpy().squeeze(0)
        whl = whl.cpu().detach().numpy().squeeze(0)

        hm = select(hm, self.conf, self.nms)

        boxes = self.pred2box(hm, offset, whl, self.conf)

        return boxes

    def pred2box(self, hm, offset, regr, thresh=0.99):
        # make binding box from heatmaps
        # thresh: threshold for logits.

        # get center
        pred = hm > thresh
        pred_center = np.where(hm > thresh)

        # get regressions (urutan sama dengan np.where: row-major)
        pred_r = regr[:, pred].T
        scores = hm[pred]

        boxes = []

        pred_center = np.asarray(pred_center).T

        for center, b, score in zip(pred_center, pred_r, scores):
            offset_xy = offset[:, center[0], center[1]]
            bbox = {
                "x": (center[1] + offset_xy[0]) * self.MODEL_SCALE,
                "y": (center[0] + offset_xy[1]) * self.MODEL_SCALE,
                "w": b[0],
                "h": b[1],
                "l": b[2],
                "score": float(score),
            }

            # discard negative values
            if bbox["w"] < 0:
                continue

            if bbox["h"] < 0:
                continue

            if bbox["l"] < 0:
                continue

            boxes.append(bbox)

        return boxes


class centernet(nn.Module):
    """
    Centernet simplified version
    Input = 1280x720 RGB image
    Backbone = ResNet-18 sampai layer4 (stride 32), jadi peta keluaran
    berukuran sekitar 23x40 (bukan 45x80). Cek dengan print(hm.shape).
    Output = heatmap utama [1,H,W], offset [2,H,W], w/h/l [3,H,W]
    """

    def __init__(self):
        super().__init__()

        # Resnet-18 as backbone.
        basemodel = torchvision.models.resnet18(weights=None)

        # Buang avgpool dan fc; sisanya sampai layer4 (512 channel, stride 32)
        self.base_model = nn.Sequential(*list(basemodel.children())[:-2])

        num_ch = 512
        head_conv = 64
        self.outc = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1, stride=1),
        )

        self.outo = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, stride=1),
        )

        self.outr = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 3, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.base_model(x)

        assert not torch.isnan(x).any()

        outc = self.outc(x)
        assert not torch.isnan(outc).any()

        outo = self.outo(x)
        assert not torch.isnan(outo).any()

        outr = self.outr(x)

        return outc, outo, outr