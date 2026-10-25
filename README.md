# Studio 2 — Klasifikasi Sampah · Kelompok 06 (Mars)

<p align="center">
<img src="https://img.shields.io/badge/Google_Colab-20232A?style=for-the-badge&logo=googlecolab&logoColor=F9AB00" alt="Google Colab"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/timm-ConvNeXt--Tiny-blue?style=for-the-badge" alt="timm ConvNeXt-Tiny"/>
</p>

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap             | NIM       | Kontribusi Utama |
|----|--------------------------|-----------|------------------|
| 1  | Bening Apni Prameswari   | 123140089 | Training dan eksperimen model |
| 2  | Raditya Alrasyid Nugroho | 123140125 | Setup repo, pipeline submission, dan training di Colab |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k` dari `timm`, 27,8 juta parameter), pretrained ImageNet-22k lalu di-fine-tune ke ImageNet-1k. Head klasifikasi diganti menjadi 5 kelas (`botol_plastik`, `gelas_plastik`, `kertas_kardus`, `saset_kemasan`, `styrofoam`). Bobot di `model.pth` disimpan dalam fp16 (± 53 MB), di bawah batas 100 MB GitHub.
- **Strategi pelatihan**:
  - Fine-tuning penuh (tanpa freezing), 10 epoch, batch 32, mixed precision fp16 pada GPU Tesla T4 di Google Colab (± 22–31 detik per epoch).
  - Optimizer AdamW dengan learning rate 2e-4 untuk seluruh layer dan weight decay 0,05.
  - Scheduler: warmup linear 1 epoch, lalu cosine decay sampai 0 (diperbarui tiap step).
  - Regularisasi: label smoothing 0,1 dan MixUp / CutMix (dipilih acak) pada 50% batch. Keduanya dipakai karena label dataset masih mengandung sedikit noise, sehingga model tidak dibuat terlalu yakin pada satu label.
  - Tanpa penyeimbangan kelas (rasio kelas terbesar/terkecil 673/365 = 1,84).
  - Augmentasi: `RandomResizedCrop` (scale 0,4–1,0), `RandomHorizontalFlip`, `TrivialAugmentWide`, `RandomErasing` (p=0,25).
- **Pemilihan model**: saat eksperimen, checkpoint yang disimpan adalah epoch dengan Macro-F1 tertinggi di `my_val/`. Setelah resepnya terbukti bagus, model untuk Preview 2 dan Final dilatih ulang dengan resep yang sama di **seluruh data train** (2.753 gambar, opsi `--full`), dan yang disimpan adalah bobot epoch terakhir (epoch 10).
- **Data tambahan**: tidak ada. Hanya dataset resmi (2.753 gambar, 5 kelas: 523 / 521 / 365 / 673 / 671 untuk `botol_plastik` / `gelas_plastik` / `kertas_kardus` / `saset_kemasan` / `styrofoam`).
- **Ukuran input & preprocessing**:
  - Training pada **256×256** (hasil `RandomResizedCrop`).
  - Inferensi (`get_transform()`): `Resize(269)` → `CenterCrop(256)` → `ToTensor` → `Normalize` (mean/std ImageNet). Ukuran gambar dan mean/std ditulis sebagai konstanta di `submission/model.py` dan dipakai juga oleh `train.py`, sehingga pengaturan training dan inferensi selalu sama.
  - TTA tidak dipakai.

## Hasil

| Putaran | Commit model | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------------|---------------------------------|------------------------------|
| Preview 1 (14 Okt) | `b0a5f71` | 0,9688 | 0,9601 |
| Preview 2 (21 Okt) | `7c86fcc` | 0,9688† | 0,9709 |
| Final (25 Okt)     | `63f40e0` | 0,9688† | Waiting |

Nilai lokal diukur dengan `evaluate.py` pada `my_val/` (413 gambar, split per kelas 85/15, `seed=42`).

† Model Preview 2 dan Final dilatih di seluruh data train, termasuk gambar `my_val/`, sehingga tidak bisa diukur langsung secara lokal (jika dijalankan, `evaluate.py` memberi 1,0000 karena gambar `my_val/` sudah ikut dilatih). Angka 0,9688 adalah skor resep yang sama saat dilatih tanpa `my_val/` (percobaan 03), dipakai sebagai perkiraan.

Sebagai pembanding, baseline ResNet-18 dari starter kit mendapat Macro-F1 sekitar 0,92 di validation set tersembunyi. Model Preview 1 kami mendapat 0,9601, lalu naik menjadi 0,9709 di Preview 2 setelah dilatih di seluruh data. Karena itu model Preview 2 dipakai juga untuk Final.

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

| # | Percobaan | Macro-F1 | Akurasi | F1 terendah | Log |
|---|-----------|----------|---------|-------------|-----|
| 02 | ConvNeXt-Tiny di CPU laptop, dihentikan di epoch 8 dari 15 (dipakai epoch 7) | 0,9628 | 0,9661 | kertas_kardus 0,933 | [02_convnext_tiny.txt](experiments/02_convnext_tiny.txt) |
| 03 | ConvNeXt-Tiny, 10 epoch di Colab T4 (dipakai epoch 5) | **0,9688** | 0,9709 | kertas_kardus 0,944 | [03_convnext_tiny_10ep_gpu.txt](experiments/03_convnext_tiny_10ep_gpu.txt) |
| 04 | ConvNeXt-Tiny, 15 epoch penuh di CPU laptop (dipakai epoch 10) | 0,9684 | 0,9709 | kertas_kardus 0,943 | [04_convnext_tiny_15ep_cpu.txt](experiments/04_convnext_tiny_15ep_cpu.txt) |
| 05 | ConvNeXt-Tiny, 10 epoch di Colab T4, seluruh data (`--full`, dipakai epoch 10) | – | – | – | [05_convnext_tiny_10ep_full.txt](experiments/05_convnext_tiny_10ep_full.txt) |

Nomor percobaan mengikuti nama file log di folder `experiments/`.

- **Split data**: 2.340 gambar untuk train dan 413 untuk validasi lokal (per kelas, proporsi kelas terjaga). Jumlah per kelas (train / val): `botol_plastik` 445 / 78, `gelas_plastik` 443 / 78, `kertas_kardus` 310 / 55, `saset_kemasan` 572 / 101, `styrofoam` 570 / 101.
- **Proses training** (percobaan 03, ± 4 menit di T4): Macro-F1 validasi naik dari 0,9038 pada epoch 1 menjadi puncak 0,9688 pada epoch 5, lalu bertahan di 0,961–0,967 sampai epoch 10. Loss turun dari 1,067 ke 0,602. Loss tidak mendekati 0 karena label smoothing dan MixUp/CutMix memang membuat target label tidak lagi 0/1.
- **10 vs 15 epoch**: percobaan 04 (15 epoch, CPU laptop, ± 8 menit per epoch) mencapai puncak 0,9684 di epoch 10, hampir sama dengan percobaan 03. Keduanya benar 401 dari 413 gambar, jadi menambah epoch tidak memberi peningkatan berarti. Karena itu 10 epoch dipakai untuk model akhir.
- **Training di seluruh data** (percobaan 05, ± 5 menit di T4): pengaturan sama dengan percobaan 03, tetapi data bertambah ± 18% (2.753 gambar) karena `my_val/` ikut dilatih. Kurva loss hampir sama dengan percobaan 03 (loss akhir 0,594 vs 0,602). Skor dari asisten naik dari 0,9601 (Preview 1) ke 0,9709 (Preview 2), yang menunjukkan tambahan data membantu.
- **Analisis kesalahan** (percobaan 03): 12 dari 413 gambar salah (akurasi 0,9709). F1 per kelas: `botol_plastik` 0,9872, `gelas_plastik` 0,9615, `kertas_kardus` 0,9444, `saset_kemasan` 0,9706, `styrofoam` 0,9802. Kelas terlemah adalah `kertas_kardus` (data paling sedikit, 365 gambar): 4 dari 55 gambarnya salah ditebak (2 sebagai `saset_kemasan`, 1 `gelas_plastik`, 1 `styrofoam`), sehingga recall-nya paling rendah. Kesalahan terbanyak adalah pertukaran `kertas_kardus` dan `saset_kemasan` (2 gambar ke tiap arah), kemungkinan karena keduanya sama-sama berupa kemasan tipis yang bisa dilipat atau diremas.
- **Waktu inferensi**: `evaluate.py` di CPU laptop (8 thread) memproses 413 gambar dalam ± 1 menit, termasuk memuat model. Untuk ± 740 gambar validation set tersembunyi diperkirakan ± 2 menit, jauh di bawah batas 15 menit.
- **Belum dicoba**: data tambahan, TTA, resolusi input selain 256 px, EMA bobot, dan learning rate berbeda untuk backbone dan head.

## Cara Reproduksi

Pelatihan dilakukan lewat notebook `colab_train.ipynb` (Google Colab, GPU Tesla T4). Notebook ini menjalankan `train.py` dari repo. Evaluasi lokal diuji dengan Python 3.11, torch 2.14, dan timm 1.0.30.

```bash
pip install torch torchvision -r submission/requirements.txt   # timm
```

1. Unduh folder `train/` dari Google Drive mata kuliah. Untuk Colab, letakkan di `MyDrive/train/<nama_kelas>/`. Untuk menjalankan di komputer sendiri, letakkan di `./train/<nama_kelas>/`. Dataset tidak disertakan di repo karena aturan lisensi melarang mengunggah ulang.
2. Jalankan notebook secara berurutan, atau jalankan langsung lewat terminal:
   ```bash
   # dengan split validasi: membuat my_train/ dan my_val/, menyimpan epoch terbaik
   python train.py --data ./train --epochs 10

   # model Final: seluruh data train, menyimpan epoch terakhir
   python train.py --data ./train --epochs 10 --full
   ```
   Keduanya menyimpan bobot ke `submission/model.pth`. Gunakan `--out <path>` untuk menyimpan ke file lain. Untuk CPU atau RAM terbatas, tambahkan `--batch-size 16 --workers 2`.
3. Evaluasi ulang secara mandiri:
   ```bash
   python evaluate.py --submission ./submission --data ./my_val
   ```

Catatan: `seed=42` dipakai untuk split dan training, tetapi training di GPU tidak sepenuhnya deterministik sehingga hasil antar-run bisa berbeda tipis.

## Struktur Repositori

```
Studio-2_Kelompok-6/
├── submission/
│   ├── model.py                  # get_model() dan get_transform()
│   ├── model.pth                 # bobot model Final (fp16, ± 53 MB)
│   └── requirements.txt          # timm
├── experiments/
│   ├── 02_convnext_tiny.txt
│   ├── 03_convnext_tiny_10ep_gpu.txt
│   ├── 04_convnext_tiny_15ep_cpu.txt
│   └── 05_convnext_tiny_10ep_full.txt
├── README.md
├── evaluate.py                   # skrip evaluasi resmi (tidak diubah)
├── train.py                      # skrip training ConvNeXt-Tiny
├── colab_train.ipynb             # notebook training di Google Colab
└── .gitignore
```

| File | Fungsi |
|------|--------|
| `submission/model.py` | Berisi `get_model()` yang membangun arsitektur ConvNeXt-Tiny dengan output 5 kelas, dan `get_transform()` untuk transform validasi. File ini tidak memuat bobot; bobot dimuat oleh `evaluate.py`. |
| `submission/model.pth` | Bobot model (state_dict) yang dinilai asisten. Berisi model hasil training di seluruh data (percobaan 05). |
| `submission/requirements.txt` | Library tambahan yang perlu dipasang, yaitu `timm`. |
| `evaluate.py` | Salinan skrip evaluasi resmi dari starter kit. Menghitung Macro-F1, akurasi, F1 per kelas, dan confusion matrix. |
| `train.py` | Membagi data menjadi `my_train/` dan `my_val/`, melatih model, mencatat loss dan Macro-F1 tiap epoch, lalu menyimpan bobot. Opsi `--full` melatih di seluruh data dan menyimpan epoch terakhir. |
| `colab_train.ipynb` | Notebook yang dipakai untuk training di Colab: mengambil dataset dari Google Drive, meng-clone repo, lalu menjalankan `train.py` dan `evaluate.py`. Output setiap sel masih tersimpan sebagai bukti hasil training. |
| `experiments/` | Log training dan evaluasi setiap percobaan. Angka di tabel catatan eksperimen diambil dari file-file ini. |
| `.gitignore` | Mencegah dataset (`train/`, `my_train/`, `my_val/`, `train_sources.csv`) dan file cache Python ikut ter-upload. |
