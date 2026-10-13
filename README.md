# Studio 2 — Klasifikasi Sampah · Kelompok 12

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Stevanus Cahya Anggara | 123140038 |                  |
| 2  | Annisa Al-Qoriah | 123140030 |                  |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ResNet-18 `torchvision` pretrained ImageNet, seluruh layer di-fine-tune; head `Linear(512, 5)`. `get_model()` sengaja dibiarkan `weights=None` (kontrak submission: arsitektur saja), bobot hasil latihan dimuat grader dari `model.pth`.
- **Strategi pelatihan**: fine-tune penuh pada holdout beku (`splits/val.txt`, 2340 train / 413 val) — AdamW `lr=3e-4`, `weight_decay=1e-4`, `CosineAnnealingLR` 15 epoch, batch 32, `CrossEntropyLoss(label_smoothing=0.1)`, augmentasi `RandomResizedCrop(224, scale=(0.6,1.0))` + `RandomHorizontalFlip` + `ColorJitter(0.3,0.3,0.3,0.05)` + `RandomErasing(p=0.25)`, early stop `patience=5`. Checkpoint terbaik dipilih dengan **macro-F1** memakai `metrics()` dan urutan kelas dari `evaluate.py` (metrik yang sama dengan penilaian asisten).
- **Data tambahan**: tidak ada — hanya dataset Studio 2 (2753 gambar di `train/`, 5 kelas).
- **Ukuran input & preprocessing**: `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`, identik untuk validasi lokal, `my_val/`, dan grader (satu sumber kebenaran: `submission/model.py`).

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | `a082efe` | 0,9373 | (menunggu asisten) |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

- Run penuh 15 epoch ResNet-18 (CPU): macro-F1 lokal **0,9373** (epoch terbaik = 11; epoch 15 = 0,9261), 150–212 s/epoch. `evaluate.py` pada `my_val` memberi angka **identik** (0,9373) dalam 23 detik untuk 413 gambar — metrik internal `train.py` sama dengan metrik grader.
- Sisa error 26 dari 413; terbanyak `kertas_kardus → styrofoam` (4) dan `kertas_kardus` sendiri adalah kelas terlemah (F1 0,9159 vs styrofoam 0,9510).
- Jadwal LR: `--epochs 1` (cosine T_max=1, LR jatuh ke 0 dalam satu epoch) sudah 0,8794, sementara epoch 1 dari run 15 epoch cuma 0,8494 (LR masih tinggi) — cosine panjang baru matang di akhir.
- Gagal: `--workers 4` → `MemoryError` di worker DataLoader (RAM 15.2 GB, sisa 2.5 GB saat itu; tiap worker ±400 MB). Fix: `--workers 2`.
- Angka lokal (413 gambar) **tidak bisa dibandingkan langsung** dengan 0,92 milik baseline starter di validation set tersembunyi (±740 gambar) — pembanding resminya kolom ketiga tabel di atas.

## Cara Reproduksi

```bash
pip install -r submission/requirements.txt
python make_split.py --check    # verifikasi holdout beku: 413 val / 2340 train, sha256 kanonik b56ef897...
python train.py                 # 15 epoch (CPU ±45 menit) -> submission/model.pth + my_val/ + runs/
python evaluate.py --submission ./submission --data ./my_val
```

`train.py` menyalin 413 nama dari `splits/val.txt` ke `my_val/` supaya skrip resmi `evaluate.py`
(bisa dibaca: `--data` harus berupa folder `<kelas>/*.jpg`) bisa dijalankan lokal.
Split **tidak** dibuat ulang saat pelatihan — `make_split.py --check` memastikan angkanya masih sebanding.

## Struktur Repositori

```
submission/
  model.py          # get_model() (arsitektur, tanpa bobot) + get_transform()
  model.pth         # state_dict model terbaik (batas teknis 500 MB; >100 MB wajib GitHub Release/LFS)
  requirements.txt  # versi paket yang dipasang asisten di environment terpisah
evaluate.py         # salinan skrip evaluasi resmi (dipakai asisten)
train.py            # pelatihan pada holdout beku splits/val.txt, tulis my_val/ + runs/
notebooks/          # train_colab.ipynb: runner Colab (clone, cek split, latih di GPU, uji, unduh submission)
make_split.py       # bikin/periksa holdout beku 15% stratified (seed 42)
train_baseline.py   # baseline dari asisten: split dibuat sendiri, checkpoint dipilih pakai akurasi
splits/val.txt      # 413 path relatif holdout beku — jangan diubah
my_val/             # tidak di-commit: salinan 413 gambar val untuk evaluate.py
runs/               # tidak di-commit: best.pth (tiap perbaikan) + last.json (metrik)
```
