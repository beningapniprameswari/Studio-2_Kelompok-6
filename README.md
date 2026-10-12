# Studio 2 — Klasifikasi Sampah · Kelompok 06

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Bening Apni Prameswari | 123140089 | Training & eksperimen model |
| 2  | Raditya Alrasyid Nugroho   | 123140125 | Setup awal repo, starter kit & pipeline submission |

## Ringkasan Pendekatan

- **Model**: ConvNeXt-Tiny dari timm (`convnext_tiny.fb_in22k_ft_in1k`), head diganti jadi 5 kelas.
- **Training**: fine-tune semua layer, AdamW (lr 2e-4, wd 0.05), warmup 1 epoch lalu cosine, 15 epoch. Epoch yang disimpan = Macro-F1 tertinggi di split validasi lokal (15% data train).
- **Augmentasi**: RandomResizedCrop (0.4–1.0), flip horizontal, TrivialAugmentWide, RandomErasing, MixUp/CutMix (p=0.5), label smoothing 0.1 karena labelnya ada yang noisy.
- **Data tambahan**: tidak ada.
- **Input**: train 256×256; validasi Resize(269) → CenterCrop(256), normalisasi ImageNet.
- **Bobot**: disimpan fp16 supaya `model.pth` di bawah 100 MB.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | | 0.9628 | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen:

- **Preview 1** — ConvNeXt-Tiny, training di CPU laptop (batch 16), dihentikan di epoch 8 dari 15; yang dipakai epoch 7. Macro-F1 my_val naik dari 0.905 (epoch 1) ke 0.963 (epoch 7). Kelas paling lemah `kertas_kardus` (F1 0.933), beberapa salah tebak ke gelas plastik, saset, dan styrofoam. Log: [experiments/02_convnext_tiny.txt](experiments/02_convnext_tiny.txt).

## Cara Reproduksi

Dataset tidak ikut di repo. Taruh folder `train/` dari Google Drive mata kuliah di root repo, lalu:

```bash
pip install torch torchvision -r submission/requirements.txt
python train.py --data ./train --epochs 15          # bikin my_val/ dan submission/model.pth
python evaluate.py --submission ./submission --data ./my_val
```

Kalau RAM terbatas (training di CPU), pakai `--batch-size 16 --workers 2`.

Untuk final, latih ulang pakai semua data train:

```bash
python train.py --data ./train --epochs 15 --full
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
