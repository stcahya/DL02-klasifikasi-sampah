"""Submission Kelompok 12 — fusi ResNet-18 + EfficientNet-B0 (late fusion) + hflip TTA.

Dua backbone dilatih terpisah, bukan dilatih bersama: melatih dua backbone bersamaan membuat
tiap cabang belajar lebih lambat (gradiennya lewat rata-rata logits) — terbukti 13 Okt:
dilatih bersama, cabang resnet 0,8666 / effnet 0,8698; ResNet-18 yang dilatih sendiri 0,9373.
Jadi checkpoint-nya digabung setelah kedua cabang selesai dilatih (experiments/fuse.py).

Bobot:
  - cabang resnet : ResNet-18 fine-tune 15 epoch di holdout beku (sama dengan commit a082efe)
  - cabang effnet : EfficientNet-B0 dari experiments/ensemble_model.py
  - gerbang GATE_RESNET = 0,75 (dari sapuan di holdout beku; lihat README)
  - inferensi memakai hflip TTA

Terukur lewat evaluate.py pada holdout beku (413 gambar):
    ResNet-18 0,9373  ->  ResNet-18+TTA 0,9403  ->  fusi ini 0,9474 (21 salah vs 24)
Biaya inferensi 0,11 s/gambar (413 gambar 45,8 detik; ±740 gambar ±1,4 menit, batas 15 menit).
Ukuran model.pth 58,3 MB (batas teknis 500 MB, batas file GitHub 100 MB).

Kontrak submission:
  - get_model() hanya MEMBANGUN arsitektur (tanpa mengunduh bobot, tanpa memuat model.pth).
  - get_transform() = transform validasi saat pelatihan, identik untuk kedua backbone
    (timm efficientnet_b0 default juga memakai mean/std ImageNet).
  - Output forward: logits [batch, 5], urutan kelas alfabetis milik evaluate.py.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms

import timm

NUM_CLASSES = 5
GATE_RESNET = 0.75  # konstanta, bukan parameter: mengubahnya tidak mengubah state_dict


class FusionTTA(nn.Module):
    """Rata-rata logits ResNet-18 (bobot 0,75) + EfficientNet-B0 (0,25) + TTA cermin."""

    def __init__(self, num_classes=NUM_CLASSES, tta=True):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.effnet = timm.create_model("efficientnet_b0", pretrained=False,
                                        num_classes=num_classes)
        self.tta = tta

    def _logits(self, x):
        return GATE_RESNET * self.resnet(x) + (1.0 - GATE_RESNET) * self.effnet(x)

    def forward(self, x):
        out = self._logits(x)
        if self.tta:
            out = 0.5 * (out + self._logits(torch.flip(x, dims=[3])))
        return out


def get_model():
    return FusionTTA()


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
