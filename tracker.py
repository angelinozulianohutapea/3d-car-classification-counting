"""
Tracking multi-objek pada bidang jalan.

Alur per frame (Tracker.update):
  1. prediksi semua track (Kalman Filter, constant velocity)
  2. asosiasi Hungarian antara track dan deteksi, dengan gerbang adaptif
     - opsional dua tahap (ByteTrack): deteksi skor rendah hanya menyambung
       track yang sudah ada, tidak boleh membuat track baru
  3. deteksi skor tinggi yang tidak cocok -> track baru
  4. hapus track: hilang terlalu lama, keluar jalur, terlalu jauh, duplikat

Rujukan: lihat config.py (Kalman 1960; Kuhn 1955 / Crouse 2016; SORT 2016;
ByteTrack 2022). Bentuk gerbang adaptif dan semua angkanya [heuristik proyek].
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

import config as cfg
from box3d import YawEstimator, blended_dims
from classifier import VehicleClassifier
from kalman_filter import KalmanFilter

IMPOSSIBLE_COST = 1000.0


class Track:
    """State satu kendaraan: Kalman Filter, klasifikasi, dan arah (yaw)."""

    def __init__(self, track_id, det, dt):
        self.id = track_id          # ID internal
        self.display_id = None      # nomor urut, diberikan saat pertama kali layak tampil
        self.age = 0
        self.hits = 1
        self.unassociated_counter = 0
        self.kf = KalmanFilter([det["x"], det["y"], det["w"], det["h"], det["l"]], dt)
        self.classifier = VehicleClassifier()
        self.yaw_estimator = YawEstimator()
        self._observe()

    # state Kalman: (x, y, vy, w, h, l)
    def x(self):
        return self.kf.estimate[0, 0]

    def y(self):
        return self.kf.estimate[1, 0]

    def w(self):
        return self.kf.estimate[3, 0]

    def h(self):
        return self.kf.estimate[4, 0]

    def l(self):
        return self.kf.estimate[5, 0]

    @property
    def cls(self):
        return self.classifier.cls

    @property
    def yaw(self):
        return self.yaw_estimator.yaw

    def dims(self):
        return blended_dims(self.classifier.cls, self.classifier.dim_buf)

    def dist(self, det):
        return float(np.hypot(self.x() - det["x"], self.y() - det["y"]))

    def update(self, det):
        self.kf.update([det["x"], det["y"], det["w"], det["h"], det["l"]])
        self.unassociated_counter = 0
        self.hits += 1
        self._observe()  # klasifikasi dan yaw hanya diperbarui saat ada deteksi nyata

    def _observe(self):
        self.classifier.update(self.w(), self.h(), self.l())
        self.yaw_estimator.update(self.x(), self.y())

    def is_visible(self):
        return (
            self.hits >= cfg.MIN_HITS_TO_DISPLAY
            and self.unassociated_counter <= cfg.MAX_LOST_TO_DISPLAY
        )

    def gate(self):
        """(jarak maks, jarak lateral maks): makin jauh makin longgar."""
        g = cfg.GATE_LOST if self.unassociated_counter > 0 else cfg.GATE_TRACKED
        depth = max(self.y(), 0.0)
        return (
            g["dist"][0] + g["dist"][1] * depth,
            g["lateral"][0] + g["lateral"][1] * depth,
        )


class Tracker:
    def __init__(self, dt):
        self.tracks = []
        self.dt = dt
        self.last_id = -1
        self.next_display_id = 1
        self._conflict_dist_sq = cfg.CONFLICT_DIST_M**2

    # ------------------------------------------------------------------
    def _add(self, det):
        self.last_id += 1
        self.tracks.append(Track(self.last_id, det, self.dt))

    def _kill(self, track_id):
        self.tracks = [t for t in self.tracks if t.id != track_id]

    def assign_display_ids(self):
        """Beri nomor urut ke track yang baru pertama kali layak tampil."""
        new = [t for t in self.tracks if t.display_id is None and t.is_visible()]
        new.sort(key=lambda t: t.y())  # serentak: yang paling dekat kamera lebih dulu
        for t in new:
            t.display_id = self.next_display_id
            self.next_display_id += 1

    def visible_tracks(self):
        return [t for t in self.tracks if t.is_visible() and t.display_id is not None]

    # ------------------------------------------------------------------
    def _match(self, track_idx, dets):
        """Hungarian dengan gerbang adaptif. Mengembalikan [(indeks track, indeks deteksi)]."""
        if not track_idx or not dets:
            return []
        cost = np.full((len(track_idx), len(dets)), IMPOSSIBLE_COST)
        for a, i in enumerate(track_idx):
            track = self.tracks[i]
            max_dist, max_lateral = track.gate()
            for j, det in enumerate(dets):
                d = track.dist(det)
                if d > max_dist or abs(det["x"] - track.x()) > max_lateral:
                    continue
                cost[a, j] = d
        rows, cols = linear_sum_assignment(cost)
        return [(track_idx[a], j) for a, j in zip(rows, cols) if cost[a, j] < IMPOSSIBLE_COST]

    def update(self, detections, conf_high=None):
        """
        detections: daftar dict {x, y, w, h, l, score} dalam meter (dunia).
        conf_high=None : satu tahap, semua deteksi setara.
        conf_high=X    : dua tahap (ByteTrack, Zhang dkk. 2022). Tahap 1: semua track
                         dengan deteksi skor >= X. Tahap 2: track yang tersisa dengan
                         deteksi skor < X. Track baru hanya dari deteksi skor tinggi.
        """
        for t in self.tracks:
            t.kf.predict()

        if conf_high is None:
            high, low = list(detections), []
        else:
            high = [d for d in detections if d.get("score", 1.0) >= conf_high]
            low = [d for d in detections if d.get("score", 1.0) < conf_high]

        if not self.tracks:
            for det in high:
                self._add(det)
            return

        n_old = len(self.tracks)
        track_matched = [False] * n_old
        high_matched = [False] * len(high)

        # tahap 1: semua track <-> deteksi skor tinggi
        for i, j in self._match(list(range(n_old)), high):
            self.tracks[i].update(high[j])
            track_matched[i] = True
            high_matched[j] = True

        # tahap 2: track yang belum cocok <-> deteksi skor rendah
        rest = [i for i, m in enumerate(track_matched) if not m]
        for i, j in self._match(rest, low):
            self.tracks[i].update(low[j])
            track_matched[i] = True

        # deteksi skor tinggi yang tidak cocok -> track baru
        for j, det in enumerate(high):
            if not high_matched[j]:
                self._add(det)

        for t in self.tracks:
            t.age += 1

        # track lama yang tidak mendapat deteksi (track baru ada di akhir daftar)
        snapshot = self.tracks.copy()
        for i, matched in enumerate(track_matched):
            if matched:
                continue
            t = snapshot[i]
            t.unassociated_counter += 1
            if not (cfg.LANE_X_MIN <= t.x() < cfg.LANE_X_MAX):
                self._kill(t.id)  # keluar jalur
            elif t.unassociated_counter >= cfg.MAX_LOST_FRAMES:
                self._kill(t.id)  # hilang terlalu lama

        for t in self.tracks.copy():
            if t.y() > cfg.MAX_TRACK_DEPTH_M:
                self._kill(t.id)

        # duplikat: pertahankan track dengan lebih banyak hit (bukti kendaraan nyata);
        # umur hanya pemutus seri. Track hantu 1-hit tidak boleh mengalahkan track nyata.
        snapshot = self.tracks.copy()
        for i, a in enumerate(snapshot):
            for b in snapshot[i + 1:]:
                d2 = (a.x() - b.x()) ** 2 + (a.y() - b.y()) ** 2
                if d2 < self._conflict_dist_sq:
                    self._kill(a.id if (a.hits, a.age) < (b.hits, b.age) else b.id)
