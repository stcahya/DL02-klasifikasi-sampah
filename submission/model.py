"""Submission Kelompok 12 — ResNet-18 pretrained ImageNet, fine-tune 15 epoch.

Ini artefak Preview 1 (commit a082efe): macro-F1 0,9373 pada holdout beku
splits/val.txt (413 gambar). Arsitekturnya sengaja tetap ResNet-18 supaya bisa
dibandingkan langsung dengan baseline resmi asisten (starter, 8 epoch, ±0,92 pada
validation set tersembunyi).

Kontrak submission:
  - get_model() hanya MEMBANGUN arsitektur (tanpa mengunduh bobot, tanpa memuat model.pth).
  - get_transform() = transform validasi saat pelatihan (Resize 256 -> CenterCrop 224,
    normalisasi ImageNet); harus identik dengan yang dipakai train.py dan grader.
  - Output forward: logits [batch, 5], urutan kelas alfabetis milik evaluate.py.
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
