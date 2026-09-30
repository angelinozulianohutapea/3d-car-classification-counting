#!/usr/bin/env python3

"""
Kalman Filter untuk tracking objek (mobil) di bidang jalan.

Asumsi:
- x konstan (mobil tidak pindah lajur)
- y bergerak dengan kecepatan konstan (vy)
- lebar, tinggi, panjang konstan

State (6):       x, y, vy, w, h, l
Measurement (5): x, y, w, h, l   (semua dalam METER, bukan pixel)

Perbaikan dibanding versi sebelumnya:
- Q kecepatan memakai percepatan maksimum * dt (bukan MAX_SPEED langsung)
- Q dimensi sangat kecil -> dimensi "terkunci", tidak membesar-mengecil
- R bergantung pada jarak objek (objek jauh = pengukuran lebih noisy)
- dt bisa diatur sesuai FPS video
"""

import numpy as np

TIME_INTERVAL = 1.0 / 30.0


class KalmanFilter:
    """
    Kalman Filter dengan model constant velocity pada sumbu y.
    """

    def __init__(self, nx, nz, first_measurement, dt=TIME_INTERVAL):
        self.nx = nx  # x, y, vy, w, h, l
        self.nz = nz  # x, y, w, h, l
        self.dt = dt
        self.K = np.zeros((nx, nz))
        assert first_measurement.shape == (nz,)

        self.estimate = np.array(
            [
                first_measurement[0],  # x
                first_measurement[1],  # y
                0.0,  # vy
                first_measurement[2],  # w
                first_measurement[3],  # h
                first_measurement[4],  # l
            ],
            dtype=float,
        ).reshape(nx, 1)

        # ------------------------------------------------------------
        # Covariance awal (dalam meter)
        # Posisi awal berasal dari pengukuran, jadi tidak perlu 100 m.
        # ------------------------------------------------------------
        self.P = np.diag(
            [
                1.0**2,  # x
                3.0**2,  # y
                20.0**2,  # vy (belum tahu kecepatan awal)
                2.0**2,  # w
                2.0**2,  # h
                3.0**2,  # l
            ]
        )
        assert np.all(self.P.T == self.P)

        # Matriks pengukuran H
        self.H = np.zeros((nz, nx))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 3] = 1
        self.H[3, 4] = 1
        self.H[4, 5] = 1

        # R awal (akan dihitung ulang tiap update sesuai jarak)
        self.R = self.measurement_noise(first_measurement[1])

        self.I = np.identity(nx)

        # ------------------------------------------------------------
        # Process noise Q  (bagian paling sensitif)
        #   terlalu kecil  -> filter lag / lambat mengikuti
        #   terlalu besar  -> filter ikut noise (kotak jumping)
        # ------------------------------------------------------------
        MAX_ACCEL = 2.0  # m/s^2, perubahan kecepatan wajar mobil di jalan tol
        LATERAL_STD = 1.0  # m/s, gerak lateral kecil (mobil jarang pindah lajur)
        DIM_STD_PER_STEP = 0.01  # m per frame -> dimensi hampir konstan

        self.Q = np.zeros((nx, nx))
        self.Q[0, 0] = (dt * LATERAL_STD) ** 2
        self.Q[1, 1] = (dt * dt * MAX_ACCEL) ** 2 + 1e-4
        self.Q[2, 2] = (dt * MAX_ACCEL) ** 2
        self.Q[3, 3] = DIM_STD_PER_STEP**2
        self.Q[4, 4] = DIM_STD_PER_STEP**2
        self.Q[5, 5] = DIM_STD_PER_STEP**2

        # Model matrix (constant velocity pada y)
        self.F = np.identity(nx)
        self.F[1, 2] = dt

        assert np.all(self.Q.T == self.Q)
        assert self.estimate.shape == (nx, 1)
        assert self.P.shape == (nx, nx)
        assert self.H.shape == (nz, nx)
        assert self.I.shape == (nx, nx)
        assert self.R.shape == (nz, nz)
        assert self.Q.shape == (nx, nx)
        assert self.K.shape == (nx, nz)

    @staticmethod
    def measurement_noise(depth):
        """
        Noise pengukuran bergantung pada jarak (depth, meter).
        - Error lateral (x) tumbuh linear terhadap jarak
        - Error longitudinal (y) tumbuh kuadratik terhadap jarak
        - Error dimensi tumbuh linear terhadap jarak
        """
        depth = max(float(depth), 0.0)

        std_x = 0.15 + 0.010 * depth
        std_y = 0.20 + 0.0005 * depth**2
        std_dim = 0.30 + 0.015 * depth

        return np.diag(
            [
                std_x**2,
                std_y**2,
                std_dim**2,
                std_dim**2,
                std_dim**2,
            ]
        )

    def predict(self):
        """
        Prediksi state dan covariance (linear)
        """
        self.estimate = self.F.dot(self.estimate)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update(self, measurement):
        """
        Update step: Kalman gain, state, covariance
        """
        assert measurement.shape == (self.nz,)

        estimate = self.estimate.reshape(self.nx, 1)

        # R disesuaikan dengan jarak pengukuran
        self.R = self.measurement_noise(measurement[1])

        # Innovation covariance
        S = self.H.dot(self.P).dot(self.H.T) + self.R

        # Kalman gain
        self.K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        assert self.K.shape == (self.nx, self.nz)

        # State update
        diff = measurement.reshape(self.nz, 1) - self.H.dot(estimate)
        self.estimate = estimate + self.K.dot(diff)

        assert self.estimate.shape == (self.nx, 1)

        # Covariance update (Joseph form, lebih stabil secara numerik)
        IKH = self.I - self.K.dot(self.H)
        self.P = IKH.dot(self.P).dot(IKH.T) + self.K.dot(self.R).dot(self.K.T)