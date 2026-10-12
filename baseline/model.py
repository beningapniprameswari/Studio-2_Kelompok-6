"""Baseline pembanding: ResNet-18 dengan head 5 kelas (salinan submission/model.py asli dari starter kit).

Dipisah dari submission/ agar train_baseline.py tidak menimpa model utama.
Uji dengan:
    python evaluate.py --submission ./baseline --data ./my_val
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5


def get_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
