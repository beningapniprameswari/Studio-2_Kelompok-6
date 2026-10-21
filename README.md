# Studio 2 — Klasifikasi Sampah · <Mars> (Kelompok 06)

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Bening Apni Prameswari   | 123140089 | Training & eksperimen model |
| 2  | Raditya Alrasyid Nugroho | 123140125 | Setup awal repo, starter kit, pipeline submission & training di Colab |

## Ringkasan Pendekatan

- **Model**: ConvNeXt-Tiny dari timm (`convnext_tiny.fb_in22k_ft_in1k`, pretrained ImageNet-22k → 1k), head diganti menjadi 5 kelas.
- **Training**: fine-tune semua layer, AdamW (lr 2e-4, weight decay 0.05), warmup 1 epoch lalu cosine decay, 10 epoch di GPU T4 (Google Colab). Saat eksperimen, bobot yang disimpan adalah epoch dengan Macro-F1 tertinggi di split validasi lokal. Untuk submission Preview 2, resep yang sama dilatih ulang di seluruh data train (`--full`, 2.753 gambar) dan yang disimpan bobot epoch terakhir (epoch 10).
- **Validasi lokal**: 15% data train dipisah per kelas (seed 42) ke `my_val/` (413 gambar); sisanya `my_train/` (2.340 gambar). Split yang sama dipakai di semua percobaan supaya hasilnya bisa dibandingkan.
- **Augmentasi**: RandomResizedCrop (skala 0.4–1.0), flip horizontal, TrivialAugmentWide, RandomErasing (p=0.25), MixUp/CutMix (p=0.5).
- **Label noisy**: label smoothing 0.1, ditambah MixUp/CutMix, supaya model tidak terlalu yakin pada label yang mungkin salah.
- **Input**: train 256×256; validasi Resize(269) → CenterCrop(256), normalisasi ImageNet.
- **Bobot**: disimpan dalam fp16 sehingga `model.pth` hanya ±53 MB (di bawah batas 100 MB GitHub).

## Data Tambahan

Tidak ada. Hanya memakai train set resmi dari Google Drive mata kuliah.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | `b0a5f71` | 0.9688 | 0.9601 |
| Preview 2 (21 Okt) | `63f40e0` | 1.0000* | |
| Final (25 Okt)     | | | |

\* Tidak valid sebagai ukuran performa: model Preview 2 dilatih di seluruh data, termasuk `my_val`. Angka ini hanya menunjukkan `model.pth` bisa dimuat dan dijalankan `evaluate.py`.

### Catatan Eksperimen

Semua diukur pada split `my_val` yang sama (413 gambar, seed 42).

| # | Percobaan | Macro-F1 | Akurasi | F1 terendah | Log |
|---|-----------|----------|---------|-------------|-----|
| 01 | Baseline ResNet-18, 8 epoch (GPU) | 0.9321 | 0.9346 | kertas_kardus 0.889 | [01_baseline_resnet18.txt](experiments/01_baseline_resnet18.txt) |
| 02 | ConvNeXt-Tiny, CPU laptop, dihentikan di epoch 8/15 (dipakai epoch 7) | 0.9628 | 0.9661 | kertas_kardus 0.933 | [02_convnext_tiny.txt](experiments/02_convnext_tiny.txt) |
| 03 | ConvNeXt-Tiny, 10 epoch di Colab T4 (dipakai epoch 5) | **0.9688** | 0.9709 | kertas_kardus 0.944 | [03_convnext_tiny_10ep_gpu.txt](experiments/03_convnext_tiny_10ep_gpu.txt) |
| 04 | ConvNeXt-Tiny, 15 epoch penuh di CPU laptop (dipakai epoch 10) | 0.9684 | 0.9709 | kertas_kardus 0.943 | [04_convnext_tiny_15ep_cpu.txt](experiments/04_convnext_tiny_15ep_cpu.txt) |
| 05 | ConvNeXt-Tiny, 10 epoch di Colab T4, seluruh data (`--full`, dipakai epoch 10) | – | – | – | [05_convnext_tiny_10ep_full.txt](experiments/05_convnext_tiny_10ep_full.txt) |

- ConvNeXt-Tiny unggul sekitar 3 poin Macro-F1 dari baseline ResNet-18. Perbaikan terbesar ada di `saset_kemasan` (0.908 → 0.971) dan `kertas_kardus` (0.889 → 0.944).
- Percobaan 03 dan 04 sama-sama benar 401/413 gambar. Setelah sekitar epoch 6, keduanya stabil di kisaran 0.961–0.968, jadi menambah epoch dari 10 ke 15 tidak memberi peningkatan berarti. Untuk Preview 1 dipakai percobaan 03.
- Percobaan 05 memakai setting percobaan 03, tapi `my_val` ikut dilatih sehingga tidak bisa diukur secara lokal. Kurva loss-nya hampir sama dengan percobaan 03 (loss akhir 0.594 vs 0.602), dan datanya ±18% lebih banyak. Model ini dipakai untuk Preview 2; skor dari asisten dibandingkan dengan Preview 1 untuk memilih model Final.
- `kertas_kardus` masih kelas terlemah. Datanya paling sedikit (±395 gambar) dan kadang tertukar dengan gelas plastik, saset, dan styrofoam.

## Cara Reproduksi

Dataset tidak disertakan di repo (sesuai aturan lisensi). Unduh folder `train/` dari Google Drive mata kuliah dan taruh di root repo.

```bash
pip install torch torchvision -r submission/requirements.txt

# training: membuat split my_train/ & my_val/, lalu menyimpan epoch terbaik ke submission/model.pth
python train.py --data ./train --epochs 10

# evaluasi dengan skrip resmi
python evaluate.py --submission ./submission --data ./my_val
```

- Kalau training di CPU atau RAM terbatas, tambahkan `--batch-size 16 --workers 2`.
- Untuk menyimpan ke file lain tanpa menimpa submission, pakai `--out <path>`.
- Untuk putaran Final, model dilatih ulang dengan seluruh data train (tanpa split validasi):

  ```bash
  python train.py --data ./train --epochs 10 --full
  ```

Training juga bisa dijalankan lewat [colab_train.ipynb](colab_train.ipynb) (Google Colab) atau [kaggle_train.ipynb](kaggle_train.ipynb).

## Struktur Repositori

```
submission/
  model.py            # get_model() dan get_transform()
  model.pth           # bobot model terbaik (fp16, ±53 MB)
  requirements.txt    # timm
evaluate.py           # salinan skrip evaluasi resmi (tidak diubah)
train.py              # training ConvNeXt-Tiny
train_baseline.py     # baseline ResNet-18 dari starter kit
baseline/model.py     # model baseline
experiments/          # log training & evaluasi tiap percobaan
colab_train.ipynb     # training di Google Colab
kaggle_train.ipynb    # training di Kaggle
```
