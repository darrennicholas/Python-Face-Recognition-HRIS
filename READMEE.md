# HR Attendance System

README singkat untuk setup environment, instalasi library, dan menjalankan aplikasi.

## 1. Persyaratan

Gunakan:

- Python 3.10
- pip
- Virtual Environment
- Webcam jika menggunakan fitur face recognition
- MySQL jika aplikasi menggunakan database MySQL

Cek versi Python:

```bash
python --version
```

Pastikan hasilnya menunjukkan:

```text
Python 3.10.x
```

> Pada Windows, jika `python --version` tidak menunjukkan Python 3.10, gunakan `py -3.10`.

---

## 2. Masuk ke Folder Project

Buka terminal lalu masuk ke folder project.

```bash
cd path/ke/project
```

Contoh:

```bash
cd C:\Users\Nama\Documents\hr-attendance-system
```

---

## 3. Membuat Virtual Environment

### Windows — Command Prompt

Buat virtual environment:

```bat
py -3.10 -m venv .venv
```

Aktifkan:

```bat
.venv\Scripts\activate
```

### Windows — PowerShell

Buat virtual environment:

```powershell
py -3.10 -m venv .venv
```

Aktifkan:

```powershell
.\.venv\Scripts\Activate.ps1
```

Jika PowerShell menolak menjalankan script, jalankan PowerShell sebagai user biasa dan gunakan:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Kemudian aktifkan kembali:

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

Buat virtual environment:

```bash
python3.10 -m venv .venv
```

Aktifkan:

```bash
source .venv/bin/activate
```

Jika berhasil, biasanya terminal akan menampilkan:

```text
(.venv)
```

---

## 4. Upgrade pip

Setelah virtual environment aktif, jalankan:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 5. Menyiapkan `requirements.txt`

Pastikan file berikut berada di folder utama project:

```text
requirements.txt
```

Isi `requirements.txt`:

```text
streamlit>=1.40,<2.0
pymysql>=1.1,<2.0
python-dotenv>=1.0,<2.0

opencv-python>=4.10,<5.0
numpy>=1.26,<2.1
Pillow>=10.0,<12.0

deepface>=0.0.100,<0.1
tensorflow>=2.15,<2.22

streamlit-webrtc>=0.77,<0.78
av>=14,<17
```

Install semua library:

```bash
python -m pip install -r requirements.txt
```

---

## 6. Library yang Digunakan

| Library | Kegunaan |
|---|---|
| `streamlit` | Membuat tampilan aplikasi web |
| `pymysql` | Koneksi ke MySQL |
| `python-dotenv` | Membaca konfigurasi dari file `.env` |
| `opencv-python` | Pemrosesan gambar dan video |
| `numpy` | Operasi numerik dan array |
| `Pillow` | Pemrosesan gambar |
| `deepface` | Face recognition |
| `tensorflow` | Backend/model machine learning untuk DeepFace |
| `streamlit-webrtc` | Akses webcam melalui WebRTC |
| `av` | Pemrosesan frame video |

---

## 7. Struktur File Dasar

Pastikan struktur project kurang lebih seperti berikut:

```text
hr-attendance-system/
│
├── app.py
├── dashboard.py
├── database.py
├── face_utils.py
├── requirements.txt
├── .env
│
├── known_faces/
│
└── .venv/
```

File dan folder dapat berbeda tergantung struktur project yang digunakan.

---

## 8. Menyiapkan File `.env`

Jika aplikasi menggunakan konfigurasi dari environment variable, buat file:

```text
.env
```

Contoh format:

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=
DB_NAME=hr_attendance
DB_PORT=3306
```

> Sesuaikan nilai tersebut dengan konfigurasi MySQL pada komputer Anda.

Jangan memasukkan password database yang sebenarnya ke repository publik.

---

## 9. Menjalankan Aplikasi

Setelah virtual environment aktif dan semua library berhasil di-install:

```bash
streamlit run app.py
```

Setelah Streamlit berjalan, buka browser pada:

```text
http://localhost:8501
```

Untuk menghentikan aplikasi:

```text
Ctrl + C
```

---

## 10. Mengecek Instalasi Library

Jalankan perintah berikut satu per satu:

```bash
python -c "import streamlit; print('Streamlit OK')"
```

```bash
python -c "import cv2; print('OpenCV OK')"
```

```bash
python -c "import numpy; print('NumPy OK')"
```

```bash
python -c "from PIL import Image; print('Pillow OK')"
```

```bash
python -c "import tensorflow; print('TensorFlow OK')"
```

```bash
python -c "import deepface; print('DeepFace OK')"
```

```bash
python -c "import streamlit_webrtc; print('WebRTC OK')"
```

```bash
python -c "import av; print('PyAV OK')"
```

```bash
python -c "import pymysql; print('PyMySQL OK')"
```

```bash
python -c "import dotenv; print('python-dotenv OK')"
```

Jika berhasil, setiap perintah akan menampilkan pesan `OK`.

---

## 11. Urutan Setup dari Awal

### Windows

```bat
cd path\ke\project
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
cd path/ke/project
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
streamlit run app.py
```

---

## 12. Saat Membuka Project Kembali

Setiap kali membuka project kembali, aktifkan virtual environment terlebih dahulu.

### Windows — Command Prompt

```bat
.venv\Scripts\activate
```

### Windows — PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Setelah aktif, jalankan aplikasi:

```bash
streamlit run app.py
```

---

## 13. Catatan

- Gunakan Python 3.10 agar sesuai dengan environment project.
- Jalankan instalasi library di dalam virtual environment.
- Pastikan webcam tersedia jika menggunakan fitur face recognition.
- Pastikan MySQL aktif jika aplikasi membutuhkan database.
- Pastikan file `.env` sudah dikonfigurasi sebelum menjalankan aplikasi.
- Jangan menghapus folder `.venv` selama masih digunakan sebagai environment project.
