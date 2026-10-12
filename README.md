# Studio 2 — Klasifikasi Sampah · Kelompok 06

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Bening Apni Prameswari | 123140089 | Training & eksperimen model |
| 2  | Raditya Alrasyid Nugroho   | 123140125 | Setup awal repo, starter kit & pipeline submission |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`, timm) pretrained ImageNet-22k → 1k, head diganti 5 kelas.
- **Strategi pelatihan**: fine-tuning seluruh layer, AdamW (lr 2e-4, weight decay 0.05), warmup 1 epoch + cosine decay, 15 epoch, mixed precision. Model dipilih berdasarkan Macro-F1 pada split validasi lokal 15%.
- **Augmentasi & regularisasi**: RandomResizedCrop (scale 0.4–1.0), horizontal flip, TrivialAugmentWide, RandomErasing, MixUp/CutMix (p=0.5), label smoothing 0.1 (membantu menghadapi label noisy).
- **Data tambahan**: tidak ada.
- **Ukuran input & preprocessing**: train 256×256; validasi Resize(269) → CenterCrop(256), normalisasi ImageNet.
- **Bobot**: disimpan sebagai `state_dict` fp16 agar `model.pth` < 100 MB (otomatis dikonversi ke fp32 saat `load_state_dict`).

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | | | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

## Cara Reproduksi

Dataset (`train/`) tidak disertakan di repo. Letakkan folder `train/` dari Google Drive mata kuliah, lalu:

```bash
pip install torch torchvision -r submission/requirements.txt
python train.py --data ./train --epochs 15          # membuat my_val/ dan submission/model.pth
python evaluate.py --submission ./submission --data ./my_val
```

Training dijalankan di Google Colab (GPU T4) melalui [colab_train.ipynb](colab_train.ipynb); alternatifnya Kaggle melalui [kaggle_train.ipynb](kaggle_train.ipynb). Log tiap eksperimen disimpan di [experiments/](experiments/).

Untuk melatih versi final di seluruh data train (tanpa split validasi):

```bash
python train.py --data ./train --epochs 15 --full
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik (fp16)
  requirements.txt  # timm
evaluate.py         # salinan skrip evaluasi resmi
train.py            # skrip training kelompok
train_baseline.py   # baseline ResNet-18 dari starter kit (menyimpan ke baseline/)
baseline/model.py   # arsitektur baseline ResNet-18, terpisah dari submission utama
experiments/        # log hasil training & evaluasi tiap eksperimen
colab_train.ipynb   # notebook untuk menjalankan training di Google Colab
kaggle_train.ipynb  # notebook untuk menjalankan training di Kaggle
```
