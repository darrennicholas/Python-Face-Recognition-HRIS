# Deploy kamera HRIS di WHM AlmaLinux lewat IP tanpa domain

Catatan implementasi 30 Oktober 2026. Konfigurasi ini berhasil menjalankan Python di VPS WHM/AlmaLinux 9 pada `/opt/face-hris` dan melayani `https://202.10.41.130:8443/camera` melalui Apache. Uji lokal menghasilkan `200` dengan `ssl_verify=0`; halaman kamera juga dapat diakses dari VPS Laravel yang terpisah. Ganti IP, domain HRIS, dan nama branch untuk instalasi baru. **Port `2087` hanya untuk WHM.**

Konfigurasi ini tidak mengubah DNS atau pointing domain yang sudah ada. Pada VPS yang diuji, port `443` dipakai banyak virtual host dan akses IP masuk ke vhost default `imitracking.kmfarma.com`. Port `8443` dipilih supaya kamera memiliki virtual host dan sertifikat IP sendiri. Perintah di bagian VPS Python dijalankan sebagai `root` di WHM Terminal.

## Status yang sudah dan belum diuji

| Bagian | Hasil sesi ini |
| --- | --- |
| Service Python, YuNet, DeepFace, ArcFace, galeri | Log `HR Attendance server READY` |
| HTTP lokal | `http://127.0.0.1:8766/camera` → `200` |
| Sertifikat IP dan HTTPS Apache | `https://202.10.41.130:8443/camera` → `200 ssl_verify=0` |
| Akses dari VPS Laravel lain | Berhasil menurut pengujian operator |
| Kamera browser, WebSocket, dan alur absensi penuh | Belum diuji dalam sesi ini |
| Pembaruan otomatis sertifikat IP | Belum dipasang; **wajib sebelum produksi** |

## 1. Siapkan repo dan branch

Repo private diambil memakai HTTPS dan GitHub personal access token **classic** dengan scope `repo`. Ketik username GitHub saat diminta; tempel token pada prompt password. Jangan taruh token di URL clone, perintah shell, atau file repo. Folder tujuan harus kosong. Titik `.` pada akhir perintah berarti isi repo masuk langsung ke `/opt/face-hris`.

```bash
mkdir -p /opt/face-hris
cd /opt/face-hris
ls -la
git clone --branch danu --single-branch https://github.com/darrennicholas/Python-Face-Recognition-HRIS.git .
git status --short --branch
git ls-files .env .env.example known_faces
```

Pada sesi ini clone pertama terlanjur mengambil `main`, lalu branch diganti menjadi `danu`. Untuk repo yang sudah ter-clone, gunakan `git fetch origin` dan `git switch --track -c danu origin/danu` setelah mencadangkan file lokal yang mungkin bertabrakan. Jangan memakai `git switch -f` atau `git reset --hard` untuk mengatasi konflik foto.

**Temuan repo pada sesi ini:** `origin/danu` memiliki `.env.example`, tetapi lima foto di `known_faces/Darren/` masih **terlacak Git**. Panduan lama yang menyebut foto sudah keluar dari Git belum sesuai dengan branch itu. Sebelum mengulang deployment, keluarkan foto dari index Git pada komputer pengembang, tambahkan `known_faces/` ke `.gitignore`, periksa hasil `git ls-files .env known_faces` kosong, lalu commit dan push. Perlakukan token lama dari riwayat `.env` sebagai terekspos dan rotasi. Cadangkan foto VPS sebelum `git switch` atau `git pull` sampai perbaikan repo selesai.

Di komputer pengembang, setelah memastikan `known_faces/` sudah ada di `.gitignore`, keluarkan foto dari **index Git saja**; file lokal tetap ada:

```bash
git rm -r --cached -- known_faces
git ls-files .env known_faces
git status --short
git add .gitignore
git diff --cached --name-status
git commit -m "Stop tracking face photos"
git push origin danu
```

Periksa daftar perubahan yang akan di-commit; jangan menyertakan file atau perubahan lain yang belum siap.

Setelah seluruh folder repo diubah kepemilikannya menjadi `facehris`, jalankan Git sebagai user itu untuk menghindari `detected dubious ownership`:

```bash
runuser -u facehris -- git -C /opt/face-hris status --short --branch
runuser -u facehris -- git -C /opt/face-hris pull --ff-only
```

## 2. Pasang Python dan dependensi

Python 3.11 dipakai tanpa mengganti Python sistem. VPS uji tidak memakai GPU NVIDIA; memasang PyTorch CPU lebih hemat ruang daripada paket CUDA. Periksa kapasitas partisi `/` sebelum instalasi karena paket AI besar.

```bash
dnf install -y python3.11 python3.11-pip mesa-libGL glib2
python3.11 --version
id facehris || useradd --system --no-create-home --home-dir /opt/face-hris --shell /sbin/nologin facehris
chown -R facehris:facehris /opt/face-hris
runuser -u facehris -- python3.11 -m venv /opt/face-hris/.venv
runuser -u facehris -- /opt/face-hris/.venv/bin/python -m pip install --upgrade pip
df -h / /tmp
runuser -u facehris -- /opt/face-hris/.venv/bin/python -m pip install --no-cache-dir 'torch==2.14.0+cpu' --index-url https://download.pytorch.org/whl/cpu
runuser -u facehris -- /opt/face-hris/.venv/bin/python -m pip install --no-cache-dir -r /opt/face-hris/requirements.txt
```

Pada VPS uji, DeepFace juga memerlukan `Werkzeug` dan, bersama TensorFlow 2.21.0, `tf-keras` 2.21.0. `pip check` saja tidak membuktikan seluruh modul bisa diimpor. Jika versi TensorFlow yang terpasang berbeda di instalasi baru, cocokkan versi `tf-keras` dengannya.

```bash
runuser -u facehris -- /opt/face-hris/.venv/bin/python -m pip install --no-cache-dir Werkzeug tf-keras==2.21.0
runuser -u facehris -- /opt/face-hris/.venv/bin/python -c "import tensorflow as tf; print(tf.__version__)"
runuser -u facehris -- /opt/face-hris/.venv/bin/python -c "from deepface import DeepFace; print('DeepFace OK')"
runuser -u facehris -- /opt/face-hris/.venv/bin/python -m pip check
```

Jika instalasi berhenti dengan `No space left on device`, periksa `df -h / /tmp`. Pada sesi ini `/` sempat hanya tersisa 791 MB; `pip cache purge` menghapus sekitar 3 GB cache dan ruang kosong naik menjadi 3,7 GB. Jangan hapus `.venv` yang sudah berisi paket.

```bash
runuser -u facehris -- /opt/face-hris/.venv/bin/python -m pip cache purge
df -h /
```

Instalasi yang terhenti saat memasang TensorFlow sempat meninggalkan paket yang tercatat di `pip show`, tetapi `import tensorflow` tidak memiliki `__version__`. Perbaikan yang berhasil pada VPS uji adalah memasang ulang **versi yang sama** tanpa mengunduh ulang seluruh dependensi:

```bash
runuser -u facehris -- /opt/face-hris/.venv/bin/python -m pip show tensorflow
runuser -u facehris -- /opt/face-hris/.venv/bin/python -m pip install --no-cache-dir --force-reinstall --no-deps tensorflow==2.21.0
```

## 3. Isi konfigurasi Python dan data wajah

Buat `.env` dari template branch `danu`. Contoh di bawah memakai HRIS pada VPS **terpisah**. Hasil `openssl rand -hex 32` dipakai sebagai rahasia yang sama di Laravel dan Python; ini **bukan** GitHub token atau Laravel `APP_KEY`. Jangan commit atau kirim nilainya ke chat.

```bash
cd /opt/face-hris
cp .env.example .env
openssl rand -hex 32
nano .env
chown facehris:facehris .env
chmod 600 .env
```

Isi yang relevan:

```dotenv
HR_CAMERA_BIND_HOST=127.0.0.1
HR_CAMERA_HTTP_PORT=8766
HR_CAMERA_WS_PORT=8765
HRIS_FACE_AUTH_EXCHANGE_URL=https://demo-hris.kmfarma.com/api/face-auth/exchange
HRIS_FACE_ATTENDANCE_EVIDENCE_URL=https://demo-hris.kmfarma.com/api/face-auth/attendance-evidence
HRIS_RETURN_URL=https://demo-hris.kmfarma.com/
FACE_RECOGNITION_TOKEN=ISI_TOKEN_ACAK_YANG_SAMA_DENGAN_LARAVEL
```

`camera_web/camera_web.js` menerima `HRIS_RETURN_URL` dari Python setelah SSO berhasil. Pastikan `.env` Python berisi URL HRIS nyata dan restart service setelah mengubahnya. Fallback JavaScript menggunakan `https://demo-hris.kmfarma.com/`.

Project Laravel yang diperiksa juga memiliki nilai mirip token di `.env.example` yang terlacak Git. Jangan pakai nilai contoh itu. Ganti dengan placeholder pada repo Laravel dan rotasi bila pernah digunakan sebagai token aktif.

Simpan foto `known_faces/` di luar `public_html`, idealnya dikirim lewat SFTP/SCP dan tidak dilacak Git. Jika direktori sudah ada, cadangkan sebelum berpindah branch. Setelah foto berada di `/opt/face-hris/known_faces`:

```bash
chown -R facehris:facehris /opt/face-hris/known_faces
chmod 700 /opt/face-hris/known_faces
find /opt/face-hris/known_faces -type f -exec chmod 600 {} +
```

## 4. Jalankan service Python

Buat `/etc/systemd/system/face-hris.service`:

```ini
[Unit]
Description=HRIS Face Recognition Camera
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=facehris
Group=facehris
WorkingDirectory=/opt/face-hris
Environment=HOME=/opt/face-hris
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/face-hris/.venv/bin/python /opt/face-hris/app.py
Restart=on-failure
RestartSec=10
TimeoutStartSec=360

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable --now face-hris
journalctl -u face-hris -f
```

Pada start pertama DeepFace mengunduh `arcface_weights.h5` sekitar 131 MB ke `/opt/face-hris/.deepface/weights/`. Tunggu `HR Attendance server READY`; `Waiting for browser connection` adalah keadaan normal. Tekan Ctrl+C untuk keluar dari `journalctl` tanpa menghentikan service. Uji:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8766/camera
systemctl status face-hris --no-pager
```

Jika service terus restart, hentikan loop sebelum membaca log: `systemctl stop face-hris` lalu `journalctl -u face-hris -n 80 --no-pager`.

## 5. Terbitkan sertifikat untuk IP tanpa mengubah domain

Periksa virtual host dan modul Apache lebih dulu. Pada VPS uji, `202.10.41.130:443` sudah dipakai banyak domain, sehingga konfigurasi kamera ditempatkan pada port `8443` yang kosong. Port `2087` milik WHM dan tidak digunakan untuk kamera.

```bash
httpd -S
httpd -M | grep -E 'proxy|ssl'
ss -ltnp | grep ':8443'
```

Jangan langsung mengedit `/etc/apache2/conf/httpd.conf`; cPanel dapat membangunnya ulang. [cPanel mendukung file `.conf` baru di `/etc/apache2/conf.d/`](https://docs.cpanel.net/ea4/apache/advanced-apache-configuration/). Untuk sertifikat IP, [Let's Encrypt memerlukan profil `shortlived`](https://letsencrypt.org/2026/03/11/shorter-certs-certbot). Certbot 5.4+ mendukung validasi IP dengan `webroot` tanpa menghentikan Apache; pada VPS uji dipakai Certbot 5.8.0 dalam virtual environment terpisah:

```bash
python3.11 -m venv /opt/certbot-ip
/opt/certbot-ip/bin/python -m pip install --no-cache-dir --upgrade pip
/opt/certbot-ip/bin/python -m pip install --no-cache-dir 'certbot>=5.4,<6'
/opt/certbot-ip/bin/certbot --version
```

`httpd -S` menunjukkan vhost default IP port 80 pada sesi ini memakai `DocumentRoot /var/www/html`. Untuk VPS lain, pastikan webroot IP-nya dulu. Uji berkas validasi lewat HTTP IP:

```bash
install -d -m 755 /var/www/html/.well-known/acme-challenge
printf 'face-hris-ip-ok\n' > /var/www/html/.well-known/acme-challenge/face-hris-ip-check
curl -sS http://202.10.41.130/.well-known/acme-challenge/face-hris-ip-check
```

Jika hasilnya `face-hris-ip-ok`, terbitkan sertifikat staging sebagai tes, kemudian sertifikat produksi. Certbot mungkin meminta email dan persetujuan syarat layanan. Jangan memasang sertifikat staging di Apache karena browser tidak mempercayainya.

```bash
/opt/certbot-ip/bin/certbot certonly --staging --cert-name face-hris-ip-staging --preferred-profile shortlived --webroot --webroot-path /var/www/html --ip-address 202.10.41.130
/opt/certbot-ip/bin/certbot certonly --cert-name face-hris-ip --preferred-profile shortlived --webroot --webroot-path /var/www/html --ip-address 202.10.41.130
```

Pada sesi ini sertifikat produksi tersimpan di `/etc/letsencrypt/live/face-hris-ip/fullchain.pem` dan kuncinya di `/etc/letsencrypt/live/face-hris-ip/privkey.pem`.

## 6. Tambahkan HTTPS Apache khusus IP port 8443

Buat `/etc/apache2/conf.d/face-hris-ip.conf` dengan isi berikut. Aturan `/ws` harus sebelum `/` karena Apache memakai proxy pertama yang cocok. Seluruh path HTTP aplikasi, termasuk `/camera_web.js`, `/camera_web.css`, dan `/register-face`, diteruskan ke port `8766`. [Rujukan ProxyPass Apache](https://httpd.apache.org/docs/2.4/mod/mod_proxy.html).

```apache
Listen 202.10.41.130:8443 https

<VirtualHost 202.10.41.130:8443>
    ServerName 202.10.41.130
    SSLEngine on
    SSLCertificateFile /etc/letsencrypt/live/face-hris-ip/fullchain.pem
    SSLCertificateKeyFile /etc/letsencrypt/live/face-hris-ip/privkey.pem

    ProxyRequests Off
    ProxyPass "/ws" "ws://127.0.0.1:8765/ws"
    ProxyPass "/" "http://127.0.0.1:8766/"
    ProxyPassReverse "/" "http://127.0.0.1:8766/"
</VirtualHost>
```

```bash
chown root:root /etc/apache2/conf.d/face-hris-ip.conf
chmod 600 /etc/apache2/conf.d/face-hris-ip.conf
httpd -t
systemctl reload httpd
ss -ltnp | grep ':8443'
curl -sS -o /dev/null -w '%{http_code} ssl_verify=%{ssl_verify_result}\n' https://202.10.41.130:8443/camera
curl -sS -o /dev/null -w '%{http_code}\n' https://202.10.41.130:8443/camera_web.js
```

Jalankan `systemctl reload httpd` hanya jika `httpd -t` menghasilkan `Syntax OK`. Hasil uji pada VPS ini: dua URL mengembalikan `200`, dan verifikasi sertifikat `0`. Dari VPS Laravel **yang terpisah**, uji akses luar:

```bash
curl -sS --connect-timeout 10 -o /dev/null -w '%{http_code} ssl_verify=%{ssl_verify_result}\n' https://202.10.41.130:8443/camera
```

Pada sesi ini tes antar-VPS berhasil. Jika tes lokal berhasil tetapi akses luar timeout, periksa firewall untuk port `8443/tcp`. Port Python `8765/8766` tetap terikat ke `127.0.0.1` dan tidak perlu dibuka ke internet.

## 7. Konfigurasi Laravel pada VPS HRIS terpisah

Kode Laravel yang diperiksa membaca `FACE_RECOGNITION_TOKEN`, `HR_CAMERA_URL`, dan `FACE_REGISTRATION_PYTHON_URL` lewat `config/services.php`. Isi `.env` **pada VPS Laravel** dengan token yang sama seperti di Python dan alamat kamera HTTPS dengan port `8443`:

```dotenv
FACE_RECOGNITION_TOKEN=ISI_TOKEN_ACAK_YANG_SAMA_DENGAN_PYTHON
HR_CAMERA_URL=https://202.10.41.130:8443/camera
FACE_REGISTRATION_PYTHON_URL=https://202.10.41.130:8443/register-face
```

Jangan gunakan `127.0.0.1` pada `FACE_REGISTRATION_PYTHON_URL` bila Laravel berada di VPS lain. Dari folder Laravel yang berisi file `artisan`, jalankan `php artisan config:cache` sebagai pemilik project. Uji kemudian login HRIS → alih ke kamera → WebSocket `wss://202.10.41.130:8443/ws` → pengenalan wajah → bukti absensi diterima Laravel → kembali ke HRIS. Alur browser penuh ini belum diuji dalam sesi dokumentasi.

## 8. Pembaruan sertifikat: wajib, belum dipasang pada sesi ini

Sertifikat IP Let's Encrypt berlaku sekitar **160 jam (enam hari)**. Pada VPS uji, sertifikat yang diterbitkan 30 Oktober 2026 kedaluwarsa 5 November 2026. Instalasi Certbot melalui virtual environment tidak otomatis membuat jadwal renewal. Langkah berikut ini adalah prosedur yang perlu diterapkan dan diuji sebelum layanan dipakai produksi; statusnya **belum terverifikasi pada VPS uji**.

Uji pembaruan melalui server staging:

```bash
/opt/certbot-ip/bin/certbot renew --dry-run --cert-name face-hris-ip
```

Lalu jadwalkan sebagai `root` dengan `crontab -e`:

```cron
0 */12 * * * /opt/certbot-ip/bin/certbot renew --quiet --cert-name face-hris-ip --deploy-hook "systemctl reload httpd"
```

`--deploy-hook` memuat ulang Apache **hanya setelah** sertifikat berhasil diperbarui. Periksa `crontab -l`, log `/var/log/letsencrypt/letsencrypt.log`, dan tanggal berakhir sertifikat secara berkala. [Panduan Certbot untuk renew dan deploy hook](https://eff-certbot.readthedocs.io/en/stable/using.html#renewing-certificates).

## Catatan operasional

- `/opt/face-hris/.cache` pernah mencapai 2,9 GB akibat instalasi pip; gunakan `--no-cache-dir` dan periksa `df -h /` sebelum mengunduh paket besar.
- `/var/log` pada VPS uji mencapai 8,8 GB, terutama `messages`, `secure`, dan `cron`. Penyebab pertumbuhan log dan rotasinya belum diselidiki. Jangan menghapus log aktif tanpa memeriksa sumbernya.
- `.cache/` dan `.lesshst` terlihat sebagai file tidak terlacak di repo VPS. Jangan jalankan `git add -A` di server.
- Sertifikat staging bernama `face-hris-ip-staging` hanya dipakai untuk uji. Jadwal renewal di atas menargetkan `face-hris-ip` produksi saja.
- Simpan backup `.env` dan `known_faces/` secara terpisah dari Git. Hingga foto dikeluarkan dari branch `danu`, `git pull` dapat mengubah foto yang dilacak.
