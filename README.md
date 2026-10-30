# HR Attendance System

Panduan VPS: [implementasi WHM lewat IP port 8443](docs/deploy-whm-almalinux-ip-8443.md), [AlmaLinux 9 + Terminal WHM + GitHub](docs/upload-github-terminal-whm-almalinux.md), atau [Ubuntu tanpa domain](docs/upload-dan-menjalankan-di-vps.md). Baca juga [rencana deployment dan batas kapasitas](docs/deployment-vps-tanpa-domain.md).

Sistem absensi karyawan berbasis **Face Recognition** menggunakan webcam browser, Python, OpenCV, YuNet, SFace, dan WebSocket.

## Requirements

- Python 3.10
- pip
- Virtual Environment
- Webcam
- Google Chrome atau Microsoft Edge

## Struktur Project

```text
hr-attendance-system/
├── app.py
├── camera_server.py
├── face_utils.py
├── requirements.txt
├── known_faces/
├── models/
└── camera_web/
    ├── camera_web.html
    ├── camera_web.css
    └── camera_web.js
```

## Requirements.txt

```txt
# Web / Server
websockets>=16.0,<17.0
python-dotenv>=1.2,<2.0

# Computer Vision
opencv-python>=4.10,<5.0
numpy>=1.26,<2.1
Pillow>=10.0,<12.0

# Face Recognition / Deep Learning
deepface>=0.0.100,<0.1
tensorflow>=2.15,<2.22
torch==2.14.0
```

## Setup

### 1. Masuk ke folder project

```bash
cd path/ke/project
```

### 2. Buat virtual environment

Windows:

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install library

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Model AI

Simpan model YuNet dan ArcFace di:

```text
models/
```

Pastikan lokasi dan nama model sesuai dengan konfigurasi di `face_utils.py`.

## Known Faces

Simpan foto wajah karyawan di:

```text
known_faces/
```

Contoh:

```text
known_faces/
├── employee_001.jpg
├── employee_002.jpg
└── employee_003.jpg
```

Pastikan foto memiliki wajah yang jelas dan sebaiknya satu foto berisi satu wajah.

## Menjalankan Aplikasi

Jalankan:

```bash
python app.py
```

Server menggunakan:

```text
HTTP      : http://127.0.0.1:8766
WebSocket : ws://127.0.0.1:8765
```

Kemudian buka browser:

```text
http://127.0.0.1:8766/camera_web.html
```

Izinkan akses webcam saat browser meminta permission.

## Arsitektur

```text
Webcam
  ↓
Browser
  ↓
camera_web.js
  ↓
WebSocket
  ↓
camera_server.py
  ↓
YuNet → Face Detection
  ↓
ArcFace → Face Recognition
  ↓
Hasil Recognition
  ↓
Browser
```

Browser menangani webcam dan tampilan bounding box, sedangkan Python menangani face detection, face recognition, dan komunikasi WebSocket.

## Troubleshooting

### Webcam tidak muncul

- Pastikan permission kamera diberikan.
- Pastikan webcam tidak digunakan aplikasi lain.
- Pastikan `app.py` sedang berjalan.
- Gunakan URL:

```text
http://127.0.0.1:8766/camera_web
```

### Wajah tidak terdeteksi

- Pastikan model YuNet tersedia.
- Pastikan path model benar.
- Pastikan wajah terlihat jelas.
- Pastikan pencahayaan cukup.

### Wajah tidak dikenali

- Pastikan model ArcFace tersedia.
- Pastikan foto wajah ada di `known_faces/`.
- Pastikan foto referensi jelas.
- Periksa konfigurasi recognition di `face_utils.py`.

## Quick Start

Windows:

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Buka:

```text
http://127.0.0.1:8766/camera_web.html
```
