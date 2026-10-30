# Upload lewat GitHub dari Terminal WHM (AlmaLinux 9)

Panduan ini mengikuti VPS `root@hcgroup` dengan **AlmaLinux 9.8, Apache/WHM**, dan folder aplikasi `/opt/face-hris`. Perintah di bagian VPS dijalankan sebagai root di **WHM » Terminal**. Jangan jalankan langkah `apt`, Nginx, atau Certbot Snap dalam [panduan Ubuntu](upload-dan-menjalankan-di-vps.md) pada server ini. Folder aplikasi berada di luar `public_html` karena `.env` dan foto wajah bukan file publik.

Panduan ini membuat kode masuk dari GitHub dan menjalankan Python sebagai service lokal. Langkah yang berhasil memakai **GitHub classic token, branch `danu`, sertifikat IP, dan Apache port 8443** dicatat di [panduan implementasi teruji](deploy-whm-almalinux-ip-8443.md). Untuk akses kamera dari browser luar melalui **IP publik + HTTPS**, Apache/SSL WHM perlu diatur sesuai virtual host dan IP yang benar.

## 1. Rapikan repo di komputer lokal sebelum push

Repo ini pernah mencatat `.env` dan foto `known_faces/` dalam Git. Pada branch `danu` yang dipakai 30 Oktober 2026, `.env` sudah tidak dilacak, tetapi lima foto `known_faces/Darren/` **masih terlacak**. Keluarkan foto dari index Git dan masukkan `known_faces/` ke `.gitignore` sebelum deployment baru. Target hasil perintah ini adalah **kosong**:

```bash
git ls-files .env known_faces
```

Periksa perubahan yang akan diunggah, lalu commit dan push dari komputer lokal:

```bash
git status --short
git add -A
git diff --cached --name-status
git commit -m "Prepare safe VPS deployment"
git push origin NAMA_BRANCH
```

Ganti `NAMA_BRANCH` dengan hasil `git branch --show-current`. Periksa `git diff --cached --name-status` sebelum commit: foto seharusnya tampil sebagai penghapusan dari index Git, `.env` tidak boleh ditambahkan, dan `.env.example` hanya boleh berisi placeholder. Jangan menambahkan `.env` atau foto kembali. Commit ini **tidak menghapus riwayat Git lama**. Jika repo pernah di-push ke GitHub, anggap kredensial di `.env` lama sudah terekspos: ganti password/token terkait dan batasi akses repo. Membersihkan riwayat Git memerlukan langkah terkoordinasi terpisah karena mengubah commit yang sudah dibagikan.

Simpan repository sebagai **private**. Bawa foto `known_faces/` ke VPS lewat SFTP/SCP/rsync, bukan GitHub. `models/face_detection_yunet_2023mar_int8.onnx` masih tercatat di Git dan akan ikut clone.

## 2. Siapkan akses GitHub dari VPS

Di Terminal WHM, periksa bahwa folder tujuan kosong:

```bash
cd /opt/face-hris
ls -la
git --version
```

Untuk repo private, buat **deploy key baca saja** di VPS:

```bash
mkdir -p /root/.ssh
chmod 700 /root/.ssh
ssh-keygen -t ed25519 -f /root/.ssh/face_hris_deploy -N '' -C 'hcgroup-face-hris'
cat /root/.ssh/face_hris_deploy.pub
```

Salin **public key** yang tercetak ke GitHub repository → **Settings → Deploy keys → Add deploy key**. Jangan aktifkan *Allow write access*. Jangan pernah menyalin isi file tanpa `.pub`. Sebelum koneksi pertama, verifikasi fingerprint host GitHub yang tampil terhadap [daftar resmi GitHub](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints).

Uji akses:

```bash
ssh -i /root/.ssh/face_hris_deploy -o IdentitiesOnly=yes -T git@github.com
```

Pesan bahwa autentikasi berhasil tetapi shell tidak disediakan oleh GitHub adalah normal. [Rujukan cPanel untuk akses repo private](https://docs.cpanel.net/knowledge-base/web-services/guide-to-git-set-up-access-to-private-repositories/).

## 3. Clone ke folder yang sudah dibuat

Jalankan dari `/opt/face-hris` **setelah** commit aman sudah ada di GitHub:

```bash
cd /opt/face-hris
GIT_SSH_COMMAND='ssh -i /root/.ssh/face_hris_deploy -o IdentitiesOnly=yes' \
  git clone git@github.com:darrennicholas/Python-Face-Recognition-HRIS.git .
git status --short
git ls-files .env known_faces
```

Titik `.` di akhir `git clone` berarti isi repo masuk ke folder saat ini. Folder harus kosong; jika gagal karena tidak kosong, periksa isinya sebelum menghapus apa pun. Target hasil `git ls-files .env known_faces` adalah kosong; pada branch `danu` saat sesi implementasi, foto masih muncul dan perlu dibersihkan dari repo.

## 4. Siapkan Python dan data runtime

Python bawaan AlmaLinux 9 adalah 3.9; project ini membutuhkan setidaknya Python 3.10 karena `torch==2.14.0`. AlmaLinux 9 menyediakan Python 3.11 sebagai paket terpisah, tanpa mengganti Python sistem. [Rujukan Red Hat tentang paket Python 3.11](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html-single/installing_and_using_dynamic_programming_languages/index), [persyaratan torch 2.14.0](https://pypi.org/project/torch/2.14.0/).

```bash
dnf install -y python3.11 python3.11-pip mesa-libGL glib2
python3.11 --version
useradd --system --no-create-home --home-dir /opt/face-hris --shell /sbin/nologin facehris
chown -R facehris:facehris /opt/face-hris
runuser -u facehris -- python3.11 -m venv /opt/face-hris/.venv
runuser -u facehris -- /opt/face-hris/.venv/bin/python -m pip install --upgrade pip
runuser -u facehris -- /opt/face-hris/.venv/bin/python -m pip install -r /opt/face-hris/requirements.txt
```

Jika `facehris` sudah pernah dibuat, abaikan perintah `useradd` dan periksa `id facehris`. Instalasi paket AI memerlukan ruang disk dan bisa lama. Jangan memakai `pip` sistem sebagai root.

Buat `.env` VPS dari template, isi URL HRIS yang nyata serta token baru yang sama dengan sisi Laravel:

```bash
cd /opt/face-hris
cp .env.example .env
nano .env
chown facehris:facehris .env
chmod 600 .env
```

`HRIS_RETURN_URL` dalam `.env` dipakai Python dan dikirim ke browser setelah SSO berhasil. JavaScript menggunakan URL tersebut untuk kembali ke HRIS; fallback di `camera_web/camera_web.js` adalah `https://demo-hris.kmfarma.com/`.

Unggah folder foto `known_faces/` secara terpisah. Jika akun cPanel `kmfa1864` memiliki akses SSH, jalankan **di komputer lokal** dari folder project (ganti `IP_VPS`):

```bash
scp -r known_faces kmfa1864@IP_VPS:/home/kmfa1864/
```

Alternatifnya, unggah folder tersebut ke `/home/kmfa1864/known_faces/` melalui SFTP; jangan gunakan `public_html`. Setelah itu, jalankan **di Terminal WHM**:

```bash
install -d -o facehris -g facehris -m 700 /opt/face-hris/known_faces
cp -a /home/kmfa1864/known_faces/. /opt/face-hris/known_faces/
```

Periksa hasil salinan sebelum menghapus salinan sementara dari akun cPanel. Atur kepemilikan dan izin:

```bash
chown -R facehris:facehris /opt/face-hris/known_faces
chmod 700 /opt/face-hris/known_faces
find /opt/face-hris/known_faces -type f -exec chmod 600 {} +
```

Jangan menaruh foto di `public_html`. Bila foto belum tersedia, buat direktori kosong dengan `install -d -o facehris -g facehris -m 700 /opt/face-hris/known_faces`; aplikasi dapat mulai, tetapi belum dapat mengenali karyawan.

## 5. Jalankan Python sebagai service

Buat `/etc/systemd/system/face-hris.service`:

```bash
nano /etc/systemd/system/face-hris.service
```

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

Tunggu `HR Attendance server READY`; tekan Ctrl+C untuk keluar dari tampilan log. Uji HTTP **lokal VPS** (hasil yang diharapkan: `200`):

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8766/camera
systemctl status face-hris
```

Jika service gagal, periksa `journalctl -u face-hris -n 100 --no-pager`. Saat pertama kali hidup, DeepFace mungkin mengunduh bobot ArcFace/Fasnet ke home user service. Pastikan VPS bisa mengakses sumber model dan user `facehris` dapat menulis ke `/opt/face-hris`.

## 6. Sebelum membuat URL publik di WHM

Apache WHM sudah menggunakan port 80/443. **Jangan** memasang Nginx kedua atau mengedit `/etc/apache2/conf/httpd.conf` langsung. cPanel dapat menimpa perubahan manual saat rebuild; perubahan virtual host memakai *include file* yang didukung cPanel. [Rujukan cPanel](https://docs.cpanel.net/ea4/apache/modify-apache-virtual-hosts-with-include-files/).

Periksa dulu pemilik virtual host IP dan modul proxy:

```bash
httpd -S
httpd -M | grep -E 'proxy|ssl'
```

Untuk halaman kamera publik, perlu HTTPS yang sertifikatnya cocok dengan **IP publik** (atau hostname yang dipakai), proxy HTTP menuju `127.0.0.1:8766`, dan proxy WebSocket `/ws` menuju `127.0.0.1:8765`. Konfigurasi Apache dan sertifikat harus dipasang pada virtual host yang benar; jika IP dibagi beberapa akun cPanel, jangan ubah vhost default tanpa memeriksa dampaknya. Pada VPS uji, `httpd -S` menunjukkan IP port 443 dipakai banyak domain; solusi tanpa mengubah pointing domain memakai HTTPS IP pada port 8443. Lihat perintah lengkap dan hasil verifikasinya di [panduan implementasi teruji](deploy-whm-almalinux-ip-8443.md). Python tetap terikat ke `127.0.0.1`, jadi port 8765/8766 tidak perlu dibuka di firewall.

## 7. Update kode sesudah deploy

Setelah perubahan terbaru di-push ke GitHub, jalankan di Terminal WHM:

```bash
cd /opt/face-hris
GIT_SSH_COMMAND='ssh -i /root/.ssh/face_hris_deploy -o IdentitiesOnly=yes' git pull --ff-only
chown -R facehris:facehris /opt/face-hris
runuser -u facehris -- /opt/face-hris/.venv/bin/python -m pip install -r /opt/face-hris/requirements.txt
systemctl restart face-hris
journalctl -u face-hris -n 100 --no-pager
```

Pada branch `danu` saat sesi implementasi, foto `known_faces/` masih dilacak Git sehingga `git pull` dapat mengubahnya. Cadangkan `known_faces/` dan `.env` secara terpisah sebelum update atau migrasi server.
