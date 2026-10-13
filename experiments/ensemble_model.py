"""KANDIDAT (bukan submission) — ensemble ResNet-18 + EfficientNet-B0 dengan hflip TTA.

Baseline resmi asisten (ResNet-18 pretrained, 8 epoch) = 0,92 macro-F1 pada validation set
tersembunyi; submission kita sekarang (ResNet-18 15 epoch) = 0,9373 pada holdout lokal.
Kandidat ini menambah keragaman prediksi: dua backbone beda keluarga (residual vs MBConv
compound-scaled) digabung dalam SATU nn.Module, plus TTA cermin saat inferensi.

Cara pakai (tanpa mengubah submission/ sedikit pun):

    python train.py --model experiments/ensemble_model.py --out runs/ens_20ep.pth --epochs 20

Kalau hasilnya > 0,9373 pada holdout yang sama, baru dipromosikan: salin file ini ke
submission/model.py BERSAMAAN dengan model.pth-nya dalam satu commit (pasangan wajib sinkron
karena evaluate.py memuat dengan strict=True).

Catatan yang sudah diverifikasi:
  - 15,2M parameter -> model.pth 58,3 MB (batas teknis 500 MB, batas file GitHub 100 MB).
  - 0,19 s/gambar di CPU dengan TTA (≈2,3 menit untuk ±740 gambar; batas 15 menit).
  - `tta` bukan bagian dari state_dict: bobot hasil latihan TTA-off tetap sahih untuk
    inferensi TTA-on (train.py mematikan TTA selama pelatihan karena melatih dengannya
    hanya menggandakan biaya).
"""
import torch
import torch.nn as nn
from torchvision import models, transforms

import timm

NUM_CLASSES = 5


class EnsembleTTA(nn.Module):
    """Rata-rata logits ResNet-18 + EfficientNet-B0, dengan TTA cermin horizontal."""

    def __init__(self, num_classes=NUM_CLASSES, tta=True):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.effnet = timm.create_model("efficientnet_b0", pretrained=False,
                                        num_classes=num_classes)
        self.tta = tta

    def _logits(self, x):
        return 0.5 * (self.resnet(x) + self.effnet(x))

    def forward(self, x):
        out = self._logits(x)
        if self.tta:
            out = 0.5 * (out + self._logits(torch.flip(x, dims=[3])))
        return out


def get_model():
    return EnsembleTTA()


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
