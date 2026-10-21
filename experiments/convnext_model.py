"""KANDIDAT (bukan submission) — ConvNeXt-Tiny dengan regularisasi modern.

Kenapa ConvNeXt-Tiny: keluarga conv murni yang dimodernisasi (depthwise 7x7, LayerNorm,
GELU) — beda keluarga dari ResNet-18 (cabang resnet) maupun EfficientNet-B0 (MBConv),
jadi cocok jadi cabang ketiga fusi atau pengganti cabang effnet yang lemah (0,8698).
Backbone pretrained-nya juga dilatih dengan resep modern (RandAugment, Mixup, stochastic
depth), jadi fitur awalnya lebih kuat daripada ResNet-18 untuk dataset kecil.

Regularisasi (bagian "optimasi" yang diminta):
  - Dropout 0,3 di head (`drop_rate`) — ConvNeXt head hanya LN -> Linear, tanpa dropout
    gampang overfit di 2340 gambar.
  - Stochastic depth 0,1 (`drop_path_rate`) — drop residual branch per-blok saat latih.
  Keduanya tidak mengubah state_dict: model.pth sahih dimuat strict=True pada arsitektur
  yang sama dengan drop dimatikan saat eval.

Optimasi lewat train.py (flag yang dipakai bersama kandidat ini):
  --randaugment   augmentasi lebih kuat dari baseline (RandAugment 2 op, magnitude 9)
  --mixup 0.2     Mixup + CutMix (alpha 0,2) + SoftTargetCrossEntropy
  --warmup 3      linear warmup 3 epoch lalu cosine decay per-batch (menstabilkan
                  fine-tune backbone yang head-nya random)

    python train.py --model experiments/convnext_model.py --out runs/convnext_30ep.pth \
        --epochs 30 --bs 32 --workers 2 --patience 30 --lr 2e-4 \
        --randaugment --mixup 0.2 --warmup 3

Kontrak sama seperti submission: get_model() hanya membangun arsitektur (pretrained=False),
get_transform() = transform validasi (Resize 256 -> CenterCrop 224, normalisasi ImageNet --
crop_pct default timm convnext_tiny = 0,875, identik dengan kandidat lain).
Wrapper kecil dipakai supaya submodule bernama `convnext` -> train.py::load_pretrained
mengenali dan menyalin bobot ImageNet-nya (nama = kunci BACKBONES di train.py).

Catatan ukuran: ConvNeXt-Tiny = 28,6M parameter -> model.pth ~109 MB (batas teknis 500 MB
OK; batas file GitHub 100 MB -> kalau dipromosikan jadi submission, lampirkan via
GitHub Release/LFS dan tulis tautannya di README).
"""
import timm
import torch.nn as nn
from torchvision import transforms

NUM_CLASSES = 5
DROP_RATE = 0.3       # dropout sebelum classifier head
DROP_PATH_RATE = 0.1  # stochastic depth di blok residual


class ConvNeXtTiny(nn.Module):
    """Wrapper agar backbone muncul sebagai submodule bernama `convnext`."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.convnext = timm.create_model(
            "convnext_tiny", pretrained=False, num_classes=num_classes,
            drop_rate=DROP_RATE, drop_path_rate=DROP_PATH_RATE)

    def forward(self, x):
        return self.convnext(x)


def get_model():
    return ConvNeXtTiny()


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
