"""Latih model dari submission/model.py pada holdout tetap splits/val.txt.

    python train.py                        # 15 epoch -> submission/model.pth
    python train.py --epochs 20 --lr 2e-4
    python train.py --scratch              # tanpa bobot ImageNet

Split TIDAK dibuat di sini. `splits/val.txt` adalah holdout beku (15%, stratified, seed 42)
yang dipakai bareng-bareng; jangan diubah, kalau nggak angka antar-anggota nggak bisa
dibandingkan.

Metrik validasi diambil dari evaluate.py, jadi angkanya identik dengan yang dihitung asisten.
Arsitektur dan transform validasi diambil dari submission/model.py -- satu sumber kebenaran.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from evaluate import CLASSES, metrics  # noqa: E402  (urutan kelas & metrik = milik grader)
from submission.model import get_model, get_transform  # noqa: E402

ROOT = Path(__file__).resolve().parent


class FileListDS(Dataset):
    """Dataset dari daftar path relatif 'kelas/nama.jpg' (isi splits/val.txt)."""

    def __init__(self, root, names, tf):
        self.root, self.names, self.tf = Path(root), names, tf

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        rel = self.names[i]
        cls = CLASSES.index(rel.split("/")[0])
        return self.tf(Image.open(self.root / rel).convert("RGB")), cls


def _imagenet_resnet18():
    return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).state_dict()


def _imagenet_effnet_b0():
    import timm  # opsional: hanya perlu kalau submission/model.py memang memakainya
    return timm.create_model("efficientnet_b0", pretrained=True).state_dict()


# nama submodule di submission/model.py -> sumber bobot ImageNet-nya
BACKBONES = {"resnet": _imagenet_resnet18, "effnet": _imagenet_effnet_b0}


def _pretrained_targets(model):
    """(prefix, modul, sumber) untuk model resnet polos, atau tiap anak yang namanya dikenal."""
    if hasattr(model, "layer4") and hasattr(model, "fc"):
        yield "", model, BACKBONES["resnet"]
    for name, mod in model.named_children():
        if name in BACKBONES:
            yield f"{name}.", mod, BACKBONES[name]


def load_pretrained(model):
    """Salin bobot ImageNet ke tiap backbone yang dikenal di dalam model; head tetap random.

    Backbone dikenali dari nama submodule (lihat BACKBONES) -- jadi arsitektur di
    submission/model.py bebas diganti/ditambah tanpa mengedit fungsi ini, asalkan
    submodule-nya bernama salah satu kunci di BACKBONES. Tensor yang bentuknya beda
    (mis. head 1000 kelas vs 5 kelas) sengaja dilewati; asersi di bawah memastikan tidak
    ada layer lain yang tertinggal random.
    """
    state, dipakai, dilewati = {}, [], []
    for prefix, mod, sumber in _pretrained_targets(model):
        own = mod.state_dict()
        for k, v in sumber().items():
            if k in own and own[k].shape == v.shape:
                state[prefix + k] = v
            else:
                dilewati.append(prefix + k)
        dipakai.append(prefix or "(model)")
    if not dipakai:
        raise SystemExit(
            f"tidak ada backbone yang dikenal di {list(dict(model.named_children()))} "
            f"-- tambahkan ke BACKBONES di train.py")
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"bobot ImageNet nggak cocok sama arsitektur: {unexpected[:3]}"
    sisa = [k for k in missing if k not in dilewati]
    assert not sisa, f"layer tak terisi: {sisa[:3]}"
    print(f"pretrained: {', '.join(dipakai)} | {len(state)} tensor dimuat | "
          f"{len(dilewati)} tensor head dilewati (random)")


def train_one(model, dl, opt, lossf, device, sched=None):
    model.train()
    tot = 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        tot += loss.item()
    return tot / len(dl)


@torch.no_grad()
def evaluate_on(model, dl, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in dl:
        y_pred += model(x.to(device)).argmax(1).cpu().tolist()
        y_true += y.tolist()
    return metrics(y_true, y_pred), y_true, y_pred


def materialize_val_dir(data, names, dst):
    """Bikin my_val/ buat evaluate.py (salin nama file, bukan bikin split baru)."""
    dst = Path(dst)
    if dst.exists():
        return dst
    for rel in names:
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(data) / rel, out)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train")
    ap.add_argument("--split", default="splits/val.txt")
    ap.add_argument("--out", default="submission/model.pth")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2,
                    help="proses loader; naikkan kalau RAM lega (tiap worker ~400MB)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5, help="berhenti kalau val tak naik N epoch")
    ap.add_argument("--scratch", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_names = [ln.strip() for ln in Path(a.split).read_text().splitlines() if ln.strip()]
    all_names = [f"{c}/{p.name}" for c in CLASSES for p in sorted((Path(a.data) / c).glob("*.jpg"))]
    val_set = set(val_names)
    tr_names = [n for n in all_names if n not in val_set]
    print(f"device={device} | train={len(tr_names)} val={len(val_names)} | kelas={CLASSES}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])
    tr_dl = DataLoader(FileListDS(a.data, tr_names, train_tf), batch_size=a.bs, shuffle=True,
                       num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    va_dl = DataLoader(FileListDS(a.data, val_names, get_transform()), batch_size=a.bs,
                       num_workers=a.workers)
    print(f"train: {len(tr_dl)} batch/epoch")

    model = get_model().to(device)
    if not a.scratch:
        load_pretrained(model)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        loss = train_one(model, tr_dl, opt, lossf, device, sched)
        m, _, _ = evaluate_on(model, va_dl, device)
        hist.append({"epoch": ep, "loss": round(loss, 4), "val_macro_f1": round(m["macro_f1"], 4),
                     "val_acc": round(m["accuracy"], 4), "sec": round(time.time() - t0, 1)})
        print(f"ep {ep:2d}/{a.epochs} loss={loss:.4f} val_macro_f1={m['macro_f1']:.4f} "
              f"val_acc={m['accuracy']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if m["macro_f1"] > best_f1:
            best_f1, bad = m["macro_f1"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # simpan tiap perbaikan: run CPU panjang gampang keinterupsi
            Path("runs").mkdir(exist_ok=True)
            torch.save(best_state, "runs/best.pth")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop: val nggak naik {a.patience} epoch")
                break

    model.load_state_dict(best_state)
    m, y_true, y_pred = evaluate_on(model, va_dl, device)
    print(f"\nBEST val_macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")
    for c, f1 in m["per_class_f1"].items():
        print(f"  {c:16s} F1={f1:.4f}")
    print("confusion (baris=true, kolom=pred):")
    print("      " + " ".join(f"{c[:7]:>7s}" for c in CLASSES))
    for c, row in zip(CLASSES, m["confusion_matrix"]):
        print(f"{c[:7]:>7s} " + " ".join(f"{v:7d}" for v in row))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.out)
    print(f"\n-> {a.out}")
    materialize_val_dir(a.data, val_names, "my_val")
    print("verifikasi: python evaluate.py --submission ./submission --data ./my_val")
    Path("runs").mkdir(exist_ok=True)
    Path("runs/last.json").write_text(json.dumps(
        {"argv": vars(a), "best_macro_f1": round(best_f1, 4), "per_class_f1": m["per_class_f1"],
         "confusion_matrix": m["confusion_matrix"], "hist": hist}, indent=2))


if __name__ == "__main__":
    main()
