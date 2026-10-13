# Studio 2 — Klasifikasi Sampah · Kelompok 06

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Bening Apni Prameswari | 123140089 | Training & eksperimen model |
| 2  | Raditya Alrasyid Nugroho   | 123140125 | Setup awal repo, starter kit & pipeline submission |

## Ringkasan Pendekatan

- **Model**: ConvNeXt-Tiny dari timm (`convnext_tiny.fb_in22k_ft_in1k`), head diganti jadi 5 kelas.
- **Training**: fine-tune semua layer, AdamW (lr 2e-4, wd 0.05), warmup 1 epoch lalu cosine, 10 epoch (GPU T4 di Colab). Epoch yang disimpan = Macro-F1 tertinggi di split validasi lokal (15% data train).
- **Augmentasi**: RandomResizedCrop (0.4–1.0), flip horizontal, TrivialAugmentWide, RandomErasing, MixUp/CutMix (p=0.5), label smoothing 0.1 karena labelnya ada yang noisy.
- **Data tambahan**: tidak ada.
- **Input**: train 256×256; validasi Resize(269) → CenterCrop(256), normalisasi ImageNet.
- **Bobot**: disimpan fp16 supaya `model.pth` di bawah 100 MB.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | | 0.9688 | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen (semua di split `my_val` yang sama, 413 gambar, seed 42):

| # | Percobaan | Macro-F1 | Akurasi | F1 terendah | Log |
|---|-----------|----------|---------|-------------|-----|
| 01 | Baseline ResNet-18, 8 epoch (GPU) | 0.9321 | 0.9346 | kertas_kardus 0.889 | [01_baseline_resnet18.txt](experiments/01_baseline_resnet18.txt) |
| 02 | ConvNeXt-Tiny, CPU laptop, dihentikan di epoch 8/15 (dipakai epoch 7) | 0.9628 | 0.9661 | kertas_kardus 0.933 | [02_convnext_tiny.txt](experiments/02_convnext_tiny.txt) |
| 03 | ConvNeXt-Tiny, 10 epoch penuh di Colab T4 (dipakai epoch 5) | **0.9688** | 0.9709 | kertas_kardus 0.944 | [03_convnext_tiny_10ep_gpu.txt](experiments/03_convnext_tiny_10ep_gpu.txt) |

- ConvNeXt-Tiny unggul ±3 poin dari baseline; perbaikan terbesar di `saset_kemasan` (0.908 → 0.971) dan `kertas_kardus` (0.889 → 0.944).
- Percobaan 03 dipakai untuk Preview 1. Epoch 6–10 stabil di 0.961–0.967, jadi 10 epoch sudah cukup.
- `kertas_kardus` masih kelas terlemah (data paling sedikit, 365 gambar); kadang tertukar dengan saset dan gelas plastik.

## Cara Reproduksi

Dataset tidak ikut di repo. Taruh folder `train/` dari Google Drive mata kuliah di root repo, lalu:

```bash
pip install torch torchvision -r submission/requirements.txt
python train.py --data ./train --epochs 10          # bikin my_val/ dan submission/model.pth
python evaluate.py --submission ./submission --data ./my_val
```

Kalau RAM terbatas (training di CPU), pakai `--batch-size 16 --workers 2`.

Untuk final, latih ulang pakai semua data train:

```bash
python train.py --data ./train --epochs 10 --full
```

Bisa juga lewat [colab_train.ipynb](colab_train.ipynb) (Google Colab) atau [kaggle_train.ipynb](kaggle_train.ipynb). Log tiap percobaan ada di [experiments/](experiments/).

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # bobot model terbaik (fp16)
  requirements.txt  # timm
evaluate.py         # skrip evaluasi resmi
train.py            # training ConvNeXt
train_baseline.py   # baseline ResNet-18 dari starter kit
baseline/model.py   # model baseline
experiments/        # log training & evaluasi
colab_train.ipynb   # training di Google Colab
kaggle_train.ipynb  # training di Kaggle
```
