# Upload dan menjalankan aplikasi di VPS tanpa domain

Panduan ini untuk **VPS Ubuntu 22.04**, akses SSH, dan IP publik tetap. Untuk **AlmaLinux + WHM**, gunakan [panduan khusus Terminal WHM dan GitHub](upload-github-terminal-whm-almalinux.md). Contoh memakai user SSH `ubuntu`, direktori aplikasi `/opt/face-hris`, dan IP `203.0.113.10`. Ganti semua nilai contoh sebelum menjalankan perintah. Laravel/HRIS sudah berjalan di alamat HTTPS yang dapat diakses VPS ini. Baca juga [rencana deployment dan batas kapasitas](deployment-vps-tanpa-domain.md).

Alur yang dipasang: browser karyawan → `https://IP_VPS/camera` → Nginx → Python HTTP `127.0.0.1:8766`; WebSocket `wss://IP_VPS/ws` → Python `127.0.0.1:8765`. Python menghubungi API HRIS. Webcam berada di perangkat karyawan, bukan VPS.

## 1. Siapkan kode sebelum upload

`camera_web/camera_web.js` sekarang memilih `ws://localhost:8765` untuk pengembangan lokal dan `wss://IP_VPS/ws` saat dibuka dari VPS. Pastikan konfigurasi reverse proxy menyediakan `/ws`.

Ganti juga `HRIS_RETURN_URL` dalam file yang sama ke alamat HTTPS HRIS yang sebenarnya. Nilai `HRIS_RETURN_URL` di `.env` Python **tidak** mengubah konstanta JavaScript ini.

`camera_server.py` sekarang memakai token hanya dari environment. Isi `FACE_RECOGNITION_TOKEN` pada `.env` VPS; tanpa token, integrasi Laravel tidak berjalan. Contoh konfigurasi kode:

```python
FACE_RECOGNITION_TOKEN = os.getenv("FACE_RECOGNITION_TOKEN", "").strip()
```

Jangan unggah `.env` lokal: isinya saat ini bukan konfigurasi deployment. Simpan foto di `known_faces/Nama Karyawan/*.jpg` dan pastikan `models/face_detection_yunet_2023mar_int8.onnx` ikut terunggah. Model ArcFace dan Fasnet dimuat DeepFace saat pertama kali dijalankan; siapkan koneksi internet VPS dan ruang disk untuk unduhan bobot model. Direktori home user service harus dapat ditulis untuk cache DeepFace.

## 2. Siapkan VPS

Login melalui SSH dari komputer Anda:

```bash
ssh ubuntu@203.0.113.10
```

Di VPS, pasang paket dasar dan buat direktori milik user `ubuntu`:

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip nginx rsync snapd libgl1 libglib2.0-0
sudo install -d -o ubuntu -g ubuntu -m 750 /opt/face-hris
sudo install -d -m 755 /var/www/letsencrypt
```

Buka TCP 80 dan 443 di firewall VPS **dan** panel firewall penyedia VPS; pertahankan akses SSH. Jika memakai UFW:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw status
```

Jika UFW belum aktif, periksa aturan sebelum `sudo ufw enable` agar koneksi SSH tidak terputus. Jangan buka port 8765/8766 ke publik.

## 3. Upload project dari komputer lokal

Jalankan perintah ini dari terminal Linux/macOS/WSL **di folder project lokal**, bukan di VPS:

```bash
rsync -av --progress \
  --exclude '.git/' --exclude '.venv/' --exclude '.env*' \
  --exclude '__pycache__/' --exclude '.streamlit/' \
  ./ ubuntu@203.0.113.10:/opt/face-hris/
```

`known_faces/` berisi foto pribadi: gunakan SSH seperti di atas, batasi akses user VPS, dan simpan cadangan terenkripsi. Untuk upload ulang kode, ulangi `rsync` tanpa `--delete` agar foto baru hasil pendaftaran di VPS tidak terhapus. Jika tidak memakai WSL, gunakan klien SFTP dan unggah file project serta folder `camera_web/`, `models/`, dan `known_faces/` ke direktori yang sama; jangan unggah `.git`, `.venv`, atau `.env` lokal.

## 4. Instal Python dan isi konfigurasi

Di VPS:

```bash
cd /opt/face-hris
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Periksa kegagalan instalasi sebelum lanjut; `requirements.txt` mencantumkan versi paket AI tertentu. Buat `.env` di VPS dengan editor, lalu isi contoh berikut memakai nilai Anda sendiri:

```bash
cd /opt/face-hris
umask 077
nano .env
chmod 600 .env
```

```dotenv
HR_CAMERA_BIND_HOST=127.0.0.1
HR_CAMERA_HTTP_PORT=8766
HR_CAMERA_WS_PORT=8765
HRIS_FACE_AUTH_EXCHANGE_URL=https://ALAMAT_HRIS/api/face-auth/exchange
HRIS_FACE_ATTENDANCE_EVIDENCE_URL=https://ALAMAT_HRIS/api/face-auth/attendance-evidence
HRIS_RETURN_URL=https://ALAMAT_HRIS/
FACE_RECOGNITION_TOKEN=TOKEN_PANJANG_ACAK_YANG_SAMA_DENGAN_LARAVEL
```

Token harus sama dengan konfigurasi Laravel. API Laravel harus dapat diakses dari VPS dengan sertifikat yang dipercaya; jika memakai CA privat, set `HRIS_FACE_CA_BUNDLE` ke path berkas CA di VPS. Jangan mematikan verifikasi TLS.

## 5. Jalankan sebagai service systemd

Buat `/etc/systemd/system/face-hris.service`:

```bash
sudo nano /etc/systemd/system/face-hris.service
```

```ini
[Unit]
Description=HRIS Face Recognition Camera
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/face-hris
Environment=HOME=/home/ubuntu
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/face-hris/.venv/bin/python /opt/face-hris/app.py
Restart=on-failure
RestartSec=10
TimeoutStartSec=360

[Install]
WantedBy=multi-user.target
```

Aktifkan dan lihat hasil preload model. Proses pertama dapat memerlukan waktu untuk mengunduh dan memuat model:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now face-hris
sudo journalctl -u face-hris -f
```

Tunggu log `HR Attendance server READY`, lalu tekan Ctrl+C untuk keluar dari tampilan log (service tetap berjalan). Periksa HTTP lokal:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8766/camera
sudo systemctl status face-hris
```

Jika service gagal, lihat `sudo journalctl -u face-hris -n 100 --no-pager`. Startup membutuhkan YuNet, ArcFace, Fasnet, dan galeri wajah; kegagalan preload menyebabkan proses keluar.

## 6. Aktifkan Nginx untuk validasi sertifikat

Buat konfigurasi HTTP awal di `/etc/nginx/sites-available/face-hris`:

```bash
sudo nano /etc/nginx/sites-available/face-hris
```

```nginx
server {
    listen 80;
    server_name 203.0.113.10;
    access_log off;

    location ^~ /.well-known/acme-challenge/ {
        root /var/www/letsencrypt;
    }

    location / { return 404; }
}
```

Aktifkan site (jika Nginx masih menampilkan halaman default, nonaktifkan site default terlebih dahulu):

```bash
sudo ln -s /etc/nginx/sites-available/face-hris /etc/nginx/sites-enabled/face-hris
sudo nginx -t
sudo systemctl reload nginx
```

Pasang Certbot melalui Snap, lalu pastikan versinya **minimal 5.4**:

```bash
sudo snap install core
sudo snap install --classic certbot
/snap/bin/certbot --version
```

Minta sertifikat untuk **IP publik milik VPS** dengan `webroot`:

```bash
sudo /snap/bin/certbot certonly --preferred-profile shortlived \
  --webroot --webroot-path /var/www/letsencrypt \
  --ip-address 203.0.113.10
```

Ikuti pertanyaan email dan persetujuan Certbot. Sertifikat IP Let's Encrypt berumur sekitar 6 hari; Certbot belum memasangnya otomatis ke Nginx, sehingga langkah berikut dan pembaruan otomatis wajib. [Rujukan resmi Let's Encrypt](https://letsencrypt.org/2026/03/11/shorter-certs-certbot).

## 7. Pasang HTTPS dan proxy WebSocket

Ganti isi `/etc/nginx/sites-available/face-hris` dengan konfigurasi ini, memakai IP yang sama dengan sertifikat:

```nginx
server {
    listen 80;
    server_name 203.0.113.10;
    access_log off;

    location ^~ /.well-known/acme-challenge/ {
        root /var/www/letsencrypt;
    }

    location / { return 301 https://$host$request_uri; }
}

server {
    listen 443 ssl;
    server_name 203.0.113.10;
    ssl_certificate /etc/letsencrypt/live/203.0.113.10/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/203.0.113.10/privkey.pem;

    # URL kamera berisi kode sekali pakai; jangan simpan query dan Referer di access log.
    access_log off;
    add_header Referrer-Policy no-referrer always;

    location /ws {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
    }

    location / {
        proxy_pass http://127.0.0.1:8766;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        client_max_body_size 5m;
    }
}
```

Nginx perlu meneruskan `Upgrade` dan `Connection` untuk WebSocket. [Rujukan resmi Nginx](https://nginx.org/en/docs/http/websocket.html). Endpoint `/register-face` juga terproksi lewat port 443; Laravel harus memanggil `https://IP_VPS/register-face` memakai token yang sama. Bila memungkinkan, batasi akses endpoint itu ke IP Laravel di firewall atau konfigurasi Nginx.

```bash
sudo nginx -t
sudo systemctl reload nginx
curl -sS -o /dev/null -w '%{http_code}\n' https://203.0.113.10/camera
```

## 8. Pastikan sertifikat diperbarui

Certbot perlu menjalankan renewal otomatis. Periksa timer/layanan renewal yang dipasang Snap dan uji simulasi:

```bash
systemctl list-timers --all | grep -i certbot
sudo /snap/bin/certbot renew --dry-run
```

Tambahkan hook agar Nginx memakai sertifikat baru sesudah renewal berhasil:

```bash
sudo install -d /etc/letsencrypt/renewal-hooks/deploy
sudo nano /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh
```

Isi file tersebut:

```sh
#!/bin/sh
systemctl reload nginx
```

```bash
sudo chmod 755 /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh
sudo /snap/bin/certbot renew --dry-run --run-deploy-hooks
```

Jika timer renewal tidak ada, perbaiki instalasi Certbot sebelum layanan dipakai. Cek tanggal kedaluwarsa secara berkala dengan `sudo /snap/bin/certbot certificates`.

## 9. Uji dari perangkat pengguna dan pemeliharaan

1. Dari jaringan lain, buka `https://IP_VPS/camera`; browser harus mempercayai sertifikat dan dapat meminta izin webcam. Halaman tanpa kode HRIS belum bisa mencatat absensi.
2. Dari HRIS, jalankan alur absensi. Pastikan pengalihan menuju `/camera?code=...`, WebSocket `wss://IP_VPS/ws` tersambung, bukti absensi diterima HRIS, lalu browser kembali ke HRIS.
3. Pastikan API HRIS bisa diakses dari VPS dan URL kamera VPS bisa diakses dari server Laravel jika pendaftaran wajah digunakan.
4. Uji restart VPS atau `sudo systemctl restart face-hris`; cek `sudo systemctl status face-hris` dan log.

Untuk upload perubahan kode kemudian, ulangi `rsync` pada langkah 3 dan jalankan:

```bash
cd /opt/face-hris
.venv/bin/python -m pip install -r requirements.txt
sudo systemctl restart face-hris
sudo journalctl -u face-hris -n 100 --no-pager
```

Cadangkan `known_faces/`, `.env`, dan bobot model/cache yang dibutuhkan sebelum migrasi VPS. Aplikasi saat ini hanya mendukung **satu sesi absensi aktif**; login baru menggantikan sesi sebelumnya. Ikuti [rencana kapasitas](deployment-vps-tanpa-domain.md#7-sebelum-melayani-100-orang-bersamaan) sebelum melayani banyak pengguna serentak.
