"""KANDIDAT (bukan submission) — EfficientNet-B0 sendirian, untuk jadi cabang kedua di fusi.

Alasan ada: melatih dua backbone bersama-sama (experiments/ensemble_model.py) membuat tiap
cabang belajar lebih lambat, karena gradiennya lewat rata-rata logits -- hasil Colab 13 Okt:
cabang resnet 0,8666 / effnet 0,8698, sementara ResNet-18 yang dilatih sendiri 0,9373.
Jalur yang dipakai: latih tiap cabang sendiri-sendiri, baru gabung logitsnya (late fusion)
lewat experiments/fuse.py.

    python train.py --model experiments/effnet_model.py --out runs/b0_25ep.pth \
        --epochs 25 --bs 32 --workers 2 --patience 25

Kontrak sama seperti submission: get_model() hanya membangun arsitektur (pretrained=False),
get_transform() = transform validasi (224 CenterCrop, normalisasi ImageNet -- sama dengan
default timm efficientnet_b0).
"""
import timm
import torch.nn as nn
from torchvision import transforms

NUM_CLASSES = 5


def get_model():
    return timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
