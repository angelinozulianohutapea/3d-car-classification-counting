# 3D Bounding Box Tracking di Jalan

Pelacakan kendaraan multi-objek dengan kotak 3D pada video jalan raya dari kamera statis (misalnya di jembatan penyeberangan). Detektor CenterNet menghasilkan kotak 2D, lalu posisi kendaraan diproyeksikan ke bidang jalan, dilacak dengan Kalman Filter, dan digambar sebagai kotak 3D lengkap dengan ID, kelas kendaraan, dan estimasi kecepatan.

## Fitur

- Deteksi kendaraan dengan model CenterNet (PyTorch, `.pth`)
- Proyeksi titik kontak ban ke bidang jalan memakai kalibrasi kamera (pitch, tinggi, focal length)
- Kalman Filter dengan noise pengukuran yang bergantung pada jarak dan dimensi kendaraan yang terkunci, sehingga kotak lebih stabil
- Matching deteksi ke objek dengan gate adaptif (objek jauh lebih longgar)
- Objek baru baru digambar setelah beberapa kali terdeteksi, supaya deteksi palsu tidak muncul
- ID tampilan berurutan (1, 2, 3, ...) dan label rapi di atas kotak (ID, kelas, km/h)
- Kotak 3D berorientasi dengan klasifikasi kelas kendaraan (car, van, dan seterusnya)
- Alat penyetel kalibrasi interaktif dengan slider (`calibrate_tuner.py`)

## Struktur Proyek

| File | Fungsi |
|---|---|
| `multi_object_tracking.py` | Program utama: deteksi, tracking, dan tampilan |
| `kalman_filter.py` | Kalman Filter (state: x, y, vy, w, h, l) |
| `object_detector.py` | Pemuatan model CenterNet dan inferensi |
| `vehicle_shape.py` | Bentuk kendaraan adaptif: kelas, dimensi, yaw, dan gambar kotak berorientasi |
| `utils.py` | Proyeksi 3D ke 2D, ray tracing ke bidang jalan, dan fungsi menggambar |
| `calibrate_tuner.py` | Alat penyetel kalibrasi kamera dengan slider |

## Instalasi

Butuh Python 3.10 atau lebih baru (diuji di Python 3.12, Ubuntu).

```bash
git clone https://github.com/angelinozulianohutapea/3d-car-classification-counting.git
cd 3d-car-classification-counting
python3 -m venv venv
source venv/bin/activate
pip install numpy opencv-python torch tqdm
```

## Menyiapkan Model dan Video

File model (`.pth`) dan video tidak disertakan di repositori karena ukurannya besar.

- Model CenterNet: `<isi tautan unduhan atau cara melatih>`
- Video contoh: `<isi sumber video>`

Taruh keduanya di folder proyek, atau gunakan path lengkapnya saat menjalankan program.

## Cara Pakai

### 1. Kalibrasi kamera

Nilai kalibrasi bawaan berasal dari simulator CARLA, jadi untuk video nyata perlu disetel ulang.

```bash
python calibrate_tuner.py VIDEO.mp4 --frame 100
```

Urutan menyetel:

1. Setel `PITCH` sampai garis horizon kuning pas dengan horizon asli.
2. Setel `FOCAL` dan `HEIGHT` sampai garis grid mengarah ke titik hilang yang sama dengan marka jalan.
3. Geser `GRID_OX` supaya garis grid menempel di marka lajur.
4. Geser `REF_X` dan `REF_Z` sampai kotak hijau berada di atas satu mobil, dengan alas menempel di aspal.
5. Cek di jarak dekat dan jauh, lalu tekan `q`.

Terminal mencetak tiga baris nilai. Salin ke `multi_object_tracking.py` (`pitch_matrix`, tinggi kamera di `to_world_from_camera`, dan `K`).

### 2. Menjalankan tracking

```bash
python multi_object_tracking.py VIDEO.mp4 PATH_MODEL.pth
```

Opsi:

| Opsi | Keterangan |
|---|---|
| `--conf` | Ambang kepercayaan deteksi (default 0.5) |
| `-f N` | Jeda visualisasi pada frame ke-N |

Tombol saat jendela tampil: `SPACE` untuk play/pause, `ESC` untuk keluar. Setiap frame hasil disimpan di folder `video_tracking_carla/`.

## Menyetel Hasil

Parameter utama ada di bagian konfigurasi `multi_object_tracking.py` dan di `kalman_filter.py`.

| Gejala | Yang disesuaikan |
|---|---|
| Kotak masih goyang | Naikkan nilai di `measurement_noise()` (`kalman_filter.py`) |
| Kotak tertinggal saat mobil mengerem atau mempercepat | Naikkan `MAX_ACCEL` |
| Ukuran kotak berubah-ubah | Turunkan `DIM_STD_PER_STEP` |
| Objek terlambat muncul | Turunkan `MIN_HITS_TO_DISPLAY` |
| ID sering berganti untuk mobil yang sama | Naikkan gate matching atau `MAX_LOST_FRAMES` |

## Keterbatasan

- Model CenterNet dilatih pada data simulasi CARLA, sehingga ada domain gap saat dipakai pada video jalan nyata. Deteksi dan ukuran kotak bisa kurang akurat.
- Kalibrasi kamera disetel manual dengan mata. Pitch, tinggi, dan focal length saling memengaruhi, jadi beberapa kombinasi bisa tampak cocok.
- Bidang jalan diasumsikan datar dan kamera statis.
- Kecepatan adalah estimasi dari hasil kalibrasi, bukan pengukuran yang sudah divalidasi.
- Klasifikasi kelas kendaraan bergantung pada dimensi terukur, sehingga ikut meleset bila kalibrasi kurang tepat.

## Kredit dan Lisensi

Proyek ini dikembangkan dari `https://github.com/antoinekeller/3d_bbox_highway.git`.

Pengembang: Angelino Zuliano Hutapea ([@angelinozulianohutapea](https://github.com/angelinozulianohutapea))
