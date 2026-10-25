"""Fine-tuning ConvNeXt-Tiny untuk klasifikasi sampah 5 kelas (Kelompok 06).

    python train.py --data ./train --epochs 10          # split 85/15, simpan epoch terbaik
    python train.py --data ./train --epochs 10 --full   # seluruh data, simpan epoch terakhir
"""
import argparse
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2

from evaluate import CLASSES, EXTS, metrics
from submission.model import IMG_SIZE, MEAN, STD, NUM_CLASSES, get_model, get_transform


def make_local_split(src, dst, frac, seed):
    """Bagi data per kelas ke my_train/ dan my_val/. Kalau my_val/ sudah ada, tidak dibuat ulang."""
    dst_tr, dst_va = Path(dst) / "my_train", Path(dst) / "my_val"
    if dst_va.exists():
        return dst_tr, dst_va
    rng = random.Random(seed)
    for cls in CLASSES:
        files = sorted(p for p in (Path(src) / cls).iterdir() if p.suffix.lower() in EXTS)
        rng.shuffle(files)
        n_va = round(len(files) * frac)
        for i, f in enumerate(files):
            out = (dst_va if i < n_va else dst_tr) / cls
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / f.name)
    return dst_tr, dst_va


def save_fp16(model, path):
    # disimpan dalam fp16 agar ukuran file di bawah batas 100 MB GitHub
    sd = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save(sd, path)


@torch.no_grad()
def evaluate_split(model, dl, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in dl:
        with torch.autocast(device, enabled=device == "cuda"):
            y_pred += model(x.to(device)).argmax(1).cpu().tolist()
        y_true += y.tolist()
    return metrics(y_true, y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder train/ dari dataset")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup", type=float, default=1.0, help="epoch warmup learning rate")
    ap.add_argument("--mix-prob", type=float, default=0.5, help="peluang batch diberi MixUp/CutMix")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--full", action="store_true", help="latih di seluruh data, tanpa validasi")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="submission/model.pth")
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if a.full:
        tr_dir, va_dir = Path(a.data), None
    else:
        tr_dir, va_dir = make_local_split(a.data, ".", a.val_frac, a.seed)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25),
    ])
    tr_ds = datasets.ImageFolder(tr_dir, train_tf)
    assert tr_ds.classes == CLASSES, f"urutan kelas tidak sesuai: {tr_ds.classes}"
    tr_dl = DataLoader(tr_ds, a.batch_size, shuffle=True, num_workers=a.workers,
                       pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    va_dl = None
    if va_dir:
        va_ds = datasets.ImageFolder(va_dir, get_transform())
        va_dl = DataLoader(va_ds, 64, num_workers=a.workers)
    print(f"device={device} | train={len(tr_ds)} | val={len(va_dl.dataset) if va_dl else 0}")

    model = get_model(pretrained=True).to(device)

    mixer = v2.RandomChoice([v2.MixUp(num_classes=NUM_CLASSES), v2.CutMix(num_classes=NUM_CLASSES)])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total = a.epochs * len(tr_dl)
    warm = int(a.warmup * len(tr_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best, best_m = -1.0, None
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if random.random() < a.mix_prob:
                x, y = mixer(x, y)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
        msg = f"epoch {ep:2d}  loss={run_loss / len(tr_dl):.4f}  ({time.time() - t0:.0f}s)"

        if va_dl:
            m = evaluate_split(model, va_dl, device)
            msg += f"  val_macroF1={m['macro_f1']:.4f}  val_acc={m['accuracy']:.4f}"
            if m["macro_f1"] > best:
                best, best_m = m["macro_f1"], m
                save_fp16(model, a.out)
                msg += "  *"
        print(msg, flush=True)

    if not va_dl:
        save_fp16(model, a.out)
        print(f"model epoch terakhir disimpan -> {a.out}")
        return
    print(f"\nterbaik val Macro-F1={best:.4f} -> {a.out}")
    for c, f in best_m["per_class_f1"].items():
        print(f"  F1 {c:<15} {f:.4f}")
    print("Confusion matrix (baris = label asli, kolom = prediksi):")
    for c, row in zip(CLASSES, best_m["confusion_matrix"]):
        print(f"  {c:<15} {row}")


if __name__ == "__main__":
    main()
