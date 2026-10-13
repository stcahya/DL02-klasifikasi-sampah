"""Submission Kelompok 12 — ResNet-18 pretrained ImageNet + hflip TTA.

Bobot = ResNet-18 hasil fine-tune 15 epoch pada holdout beku splits/val.txt (commit a082efe),
tanpa perubahan. Yang ditambahkan cuma TTA cermin horizontal saat inferensi: logits diambil
dari gambar asli dan gambar terbalik kiri-kanan, lalu dirata-rata.

Alasan dipakai (terukur di holdout beku, 413 gambar, metrik evaluate.py):
    tanpa TTA 0,9373  ->  dengan TTA 0,9403
Selisih ini kecil (2 gambar dari 413, ambang noise), tapi arahnya konsisten dengan latihan
yang memakai RandomHorizontalFlip, dan biayanya cuma 2x forward (413 gambar: 15 s -> 25 s;
validation set tersembunyi ±740 gambar tetap jauh di bawah batas 15 menit).

Kontrak submission:
  - get_model() hanya MEMBANGUN arsitektur (tanpa mengunduh bobot, tanpa memuat model.pth).
  - get_transform() = transform validasi saat pelatihan (Resize 256 -> CenterCrop 224,
    normalisasi ImageNet), sama dengan yang dipakai train.py dan grader.
  - Output forward: logits [batch, 5], urutan kelas alfabetis milik evaluate.py.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5


class ResNet18TTA(models.ResNet):
    """ResNet-18 dengan TTA cermin horizontal.

    Sengaja mewarisi models.ResNet (bukan membungkus `self.net = models.resnet18(...)`),
    supaya nama key state_dict tetap `conv1.*`/`bn1.*`/`fc.*` persis seperti resnet18 biasa --
    artinya model.pth hasil latihan lama tetap termuat apa adanya dengan strict=True, tanpa
    remap key.
    """

    def __init__(self, num_classes=NUM_CLASSES, tta=True):
        super().__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2],
                         num_classes=num_classes)
        self.tta = tta

    def forward(self, x):
        out = super().forward(x)
        if self.tta:
            out = 0.5 * (out + super().forward(torch.flip(x, dims=[3])))
        return out


def get_model():
    return ResNet18TTA()


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
