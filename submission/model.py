"""Model submission Kelompok 06: ConvNeXt-Tiny dari timm dengan output 5 kelas.

Urutan kelas: 0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
"""
import timm
from torchvision import transforms

NUM_CLASSES = 5
# konstanta di bawah juga dipakai oleh train.py
MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"
IMG_SIZE = 256
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_model(pretrained=False):
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=NUM_CLASSES)


def get_transform():
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE / 0.95)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
