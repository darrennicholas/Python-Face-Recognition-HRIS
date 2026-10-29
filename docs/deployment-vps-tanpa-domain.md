# Rencana deployment di VPS tanpa domain

Langkah upload lewat GitHub dari Terminal WHM tersedia di [panduan AlmaLinux + WHM](upload-github-terminal-whm-almalinux.md). Untuk Ubuntu, lihat [panduan Nginx dan sertifikat IP](upload-dan-menjalankan-di-vps.md).

Proyek ini dapat diakses melalui **IP publik VPS**; domain khusus untuk layanan Python tidak wajib. Contoh alamat halaman kamera: `https://203.0.113.10/camera` (`203.0.113.10` hanya contoh, ganti dengan IP VPS). Webcam tetap berada di perangkat karyawan dan frame dikirim dari browser ke Python melalui WebSocket.

> Dokumen ini adalah daftar pekerjaan deployment, bukan tanda bahwa kode sekarang sudah siap untuk 100 pengguna bersamaan.

## 1. Pastikan alur Laravel ↔ Python dapat dijangkau

1. Laravel mengarahkan browser karyawan ke `https://IP_VPS/camera?code=KODE_SEKALI_PAKAI`.
2. Browser membuka webcam dan mengirim frame ke layanan Python melalui `wss://IP_VPS/ws`.
3. Python menukar kode tersebut ke API Laravel (`HRIS_FACE_AUTH_EXCHANGE_URL`), lalu mengirim bukti absensi ke `HRIS_FACE_ATTENDANCE_EVIDENCE_URL`.
4. Setelah selesai, browser kembali ke `HRIS_RETURN_URL`.

Laravel boleh mempunyai domain sendiri walaupun VPS Python hanya mempunyai IP. Pastikan API Laravel dapat diakses **dari VPS Python**. Alamat bawaan `hris-medialys.test` pada [camera_server.py](../camera_server.py) dan [camera_web.js](../camera_web/camera_web.js) adalah alamat pengembangan; ganti dengan alamat HRIS yang benar atau sediakan koneksi jaringan privat yang dapat menjangkaunya. Jika Laravel memanggil endpoint pendaftaran wajah Python, arah panggilannya juga harus dapat menjangkau VPS Python.

## 2. Siapkan VPS dan aplikasi

- Gunakan IP publik yang stabil. Buka port **80** untuk validasi dan pembaruan sertifikat serta **443** untuk pengguna. Batasi SSH ke administrator.
- Pasang Python yang kompatibel dengan proyek (README saat ini menyebut Python 3.10), buat virtual environment, lalu instal `requirements.txt`.
- Salin kode, model di `models/`, serta data wajah di `known_faces/` melalui jalur aman. Perlakukan foto wajah dan token sebagai data sensitif; siapkan cadangannya.
- Jalankan `python app.py` melalui pengelola layanan seperti systemd agar otomatis hidup kembali setelah VPS reboot. Uji bahwa model YuNet, ArcFace, dan Fasnet berhasil dimuat sebelum membuka layanan ke pengguna.
- Biarkan `HR_CAMERA_BIND_HOST=127.0.0.1` dan port internal bawaan **8766 (HTTP)** serta **8765 (WebSocket)**. Nginx akan menjadi pintu publik; port internal ini tidak perlu dibuka di firewall.

## 3. Aktifkan HTTPS pada IP publik

Browser memerlukan *secure context* untuk `getUserMedia()`. `http://IP_VPS` tidak memenuhi syarat tersebut; pengecualian `http://localhost` hanya berlaku pada perangkat lokal. Gunakan sertifikat tepercaya untuk IP publik dan akses halaman lewat `https://IP_VPS`. [Rujukan MDN](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

Let's Encrypt sudah menerbitkan sertifikat IP publik. Dengan Certbot **5.4 atau lebih baru**, salah satu cara adalah metode `webroot` berikut setelah Nginx melayani `/.well-known/acme-challenge/` dari direktori yang dipilih:

```bash
sudo certbot certonly --preferred-profile shortlived \
  --webroot --webroot-path /var/www/letsencrypt \
  --ip-address IP_VPS
```

Konfigurasikan Nginx secara manual memakai sertifikat dan kunci di `/etc/letsencrypt/live/IP_VPS/`. Sertifikat IP ini berlaku sekitar **6 hari**, sehingga pembaruan otomatis, pemuatan ulang Nginx setelah pembaruan, dan pengujian `certbot renew --dry-run` wajib disiapkan. Certbot tidak otomatis memasang sertifikat IP ke konfigurasi Nginx. [Petunjuk resmi Let's Encrypt](https://letsencrypt.org/2026/03/11/shorter-certs-certbot).

## 4. Teruskan HTTP dan WebSocket melalui Nginx

- `https://IP_VPS/camera` dan aset halaman kamera → `http://127.0.0.1:8766`.
- `wss://IP_VPS/ws` → `http://127.0.0.1:8765`; konfigurasi proxy harus meneruskan header `Upgrade` dan `Connection` untuk WebSocket. [Rujukan Nginx](https://nginx.org/en/docs/http/websocket.html).
- Pertahankan jalur `/.well-known/acme-challenge/` pada port 80 untuk pembaruan sertifikat; arahkan permintaan HTTP lainnya ke HTTPS.

`websocketURL` pada [camera_web.js](../camera_web/camera_web.js) sekarang menggunakan alamat WSS pada IP VPS saat halaman dibuka dari server. Alamat `127.0.0.1` di JavaScript pengguna menunjuk ke **perangkat pengguna**, bukan VPS. Sesuaikan URL kembali ke HRIS yang masih memakai `.test` pada file itu. URL lokal yang dicetak [app.py](../app.py) hanya pesan di terminal; URL yang dibagikan kepada pengguna adalah alamat HTTPS reverse proxy.

## 5. Isi konfigurasi dan amankan integrasi

Setel variabel berikut di `.env` pada VPS sesuai lingkungan sebenarnya:

```dotenv
HR_CAMERA_BIND_HOST=127.0.0.1
HRIS_FACE_AUTH_EXCHANGE_URL=https://ALAMAT_HRIS/api/face-auth/exchange
HRIS_FACE_ATTENDANCE_EVIDENCE_URL=https://ALAMAT_HRIS/api/face-auth/attendance-evidence
HRIS_RETURN_URL=https://ALAMAT_HRIS/
FACE_RECOGNITION_TOKEN=TOKEN_RAHASIA_BARU
```

Gunakan token yang sama pada sisi Laravel, batasi izin baca `.env`, dan rotasi token lama. [camera_server.py](../camera_server.py) kini tidak mempunyai nilai token bawaan. Pastikan sertifikat HTTPS API Laravel valid dari VPS Python.

## 6. Uji alur dari perangkat lain

1. Dari laptop/ponsel di jaringan lain, buka `https://IP_VPS/camera` dan pastikan sertifikat dipercaya browser serta izin webcam muncul.
2. Masuk melalui HRIS Laravel; pastikan pengalihan dengan kode sekali pakai berhasil, WebSocket tersambung melalui `wss://`, wajah dikenali, bukti absensi diterima Laravel, dan browser kembali ke HRIS.
3. Uji juga kegagalan: kode kedaluwarsa, wajah tidak cocok, koneksi terputus, dan layanan hidup kembali setelah reboot.
4. Periksa pembaruan sertifikat otomatis dan log aplikasi/Nginx tanpa menampilkan token atau foto wajah secara terbuka.

## 7. Sebelum melayani 100 orang bersamaan

Kode sekarang hanya mempunyai **satu sesi absensi aktif**: login berikutnya menggantikan sesi sebelumnya ([camera_server.py](../camera_server.py)). Membesarkan VPS tidak menyelesaikan batas ini. Pisahkan status autentikasi, frame, hasil pengenalan, dan pekerja pemrosesan per sesi; setelah itu uji beban bertahap.

Browser saat ini dapat mengirim sampai **25 frame per detik per orang** ([camera_web.js](../camera_web/camera_web.js)). Pada 100 orang, batas teoritis kiriman adalah 2.500 frame/detik. Kurangi frekuensi frame atau gunakan alur pengambilan frame seperlunya sebelum menentukan kapasitas produksi. Perkiraan awal untuk **uji beban** setelah perubahan tersebut adalah 16–32 vCPU dan 32–64 GB RAM; ini bukan jaminan bahwa satu VPS sanggup melayani 100 sesi. Ukur latensi absensi, CPU, RAM, dan jaringan pada 10, 25, 50, lalu 100 sesi; bila perlu, gunakan GPU atau beberapa server.
