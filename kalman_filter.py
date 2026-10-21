"""
Kalman Filter untuk melacak kendaraan pada bidang jalan.

State (6):       x, y, vy, w, h, l
Measurement (5): x, y, w, h, l   (semua dalam METER)

Asumsi: x konstan (jarang pindah lajur), y bergerak dengan kecepatan konstan
(vy), dimensi konstan. vy WAJIB ada di state karena dipakai untuk
memprediksi posisi di frame berikutnya; yang dihapus hanya tampilannya.

Dasar: Kalman (1960); constant velocity dan bentuk Joseph untuk update
covariance: Bar-Shalom dkk. (2001). Lihat config.py untuk besaran Q dan R.
"""

import numpy as np

import config as cfg

TIME_INTERVAL_FALLBACK = 1.0 / 30.0  # dipakai bila FPS video tidak terbaca


def measurement_noise(depth):
    """Noise pengukuran membesar dengan jarak [heuristik, lihat config.py]."""
    depth = max(float(depth), 0.0)

    def std(par):
        a, b, p = par
        return a + b * depth**p

    return np.diag([std(cfg.KF_STD_X) ** 2, std(cfg.KF_STD_Y) ** 2] + [std(cfg.KF_STD_DIM) ** 2] * 3)


class KalmanFilter:
    NX = 6
    NZ = 5

    def __init__(self, first_measurement, dt):
        z = np.asarray(first_measurement, dtype=float).reshape(-1)
        assert z.shape == (self.NZ,)
        self.dt = dt
        self.estimate = np.array([z[0], z[1], 0.0, z[2], z[3], z[4]]).reshape(self.NX, 1)
        self.P = np.diag(np.square(cfg.KF_INIT_STD))

        # matriks pengukuran: z = (x, y, w, h, l) dari state (x, y, vy, w, h, l)
        self.H = np.zeros((self.NZ, self.NX))
        for row, col in enumerate((0, 1, 3, 4, 5)):
            self.H[row, col] = 1.0

        # constant velocity pada y
        self.F = np.identity(self.NX)
        self.F[1, 2] = dt

        self.Q = np.zeros((self.NX, self.NX))
        self.Q[0, 0] = (dt * cfg.LATERAL_STD) ** 2
        self.Q[1, 1] = (dt * dt * cfg.MAX_ACCEL) ** 2 + 1e-4
        self.Q[2, 2] = (dt * cfg.MAX_ACCEL) ** 2
        for i in (3, 4, 5):
            self.Q[i, i] = cfg.DIM_STD_PER_STEP**2

        self.I = np.identity(self.NX)

    def predict(self):
        self.estimate = self.F.dot(self.estimate)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update(self, measurement):
        z = np.asarray(measurement, dtype=float).reshape(self.NZ, 1)
        R = measurement_noise(z[1, 0])  # R bergantung pada jarak pengukuran
        S = self.H.dot(self.P).dot(self.H.T) + R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.estimate = self.estimate + K.dot(z - self.H.dot(self.estimate))
        IKH = self.I - K.dot(self.H)
        self.P = IKH.dot(self.P).dot(IKH.T) + K.dot(R).dot(K.T)  # bentuk Joseph
