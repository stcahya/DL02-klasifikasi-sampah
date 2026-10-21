# Studio 2 — Klasifikasi Sampah · Kelompok 13/Pomelo

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Stevanus Cahya Anggara | 123140038 |                  |
| 2  | Annisa Al-Qoriah | 123140030 |                  |

## Ringkasan Pendekatan

- **Arsitektur (Preview 2 — ConvNeXt-Tiny)**: backbone tunggal timm `convnext_tiny` (pretrained ImageNet, fine-tune penuh), head `Linear(768→5)` dengan **dropout 0,3** (`drop_rate`) dan **stochastic depth 0,1** (`drop_path_rate`) saat latih — keduanya nonaktif saat eval sehingga `model.pth` sahih dimuat `strict=True`. Menggantikan late fusion ResNet-18 + EfficientNet-B0 milik Preview 1 (0,9474) karena sendirian mencapai **0,9762** di holdout beku yang sama.
- **Strategi pelatihan**: fine-tune penuh pada holdout beku (`splits/val.txt`, 2340 train / 413 val) — AdamW `lr=2e-4`, 30 epoch, batch 32; **RandAugment** (2 op, magnitude 9) + crop agresif + erasing p=0,5; **Mixup/CutMix α=0,2** dengan `SoftTargetCrossEntropy`; **linear warmup 3 epoch** lalu cosine decay per batch. Checkpoint terbaik dipilih dengan **macro-F1** memakai `metrics()` dan urutan kelas dari `evaluate.py` (metrik yang sama dengan penilaian asisten).
- **Data tambahan**: tidak ada — hanya dataset Studio 2 (2753 gambar di `train/`, 5 kelas).
- **Ukuran input & preprocessing**: `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])` — satu sumber kebenaran: `submission/model.py`.
- **Ukuran & waktu**: `model.pth` 111,4 MB (28,6M parameter) → **didistribusikan lewat GitHub Release** (bukan git, batas 100 MB); 413 gambar 262 detik di CPU → ±740 gambar ≈ 8 menit (batas asisten 15 menit).
- **Riwayat Preview 1 (late fusion)**: ResNet-18 + EfficientNet-B0 dilatih **terpisah**, logits digabung `0,75/0,25` + hflip TTA dalam satu `nn.Module` → 0,9474. Melatih dua backbone bersamaan membuat tiap cabang belajar lebih lambat (gradien lewat rata-rata logits): 13 Okt cabangnya hanya 0,8666/0,8698 vs ResNet-18 sendirian 0,9373. Arsitektur & resepnya dipertahankan di `experiments/ensemble_model.py` + `experiments/fuse.py`.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | `e2a1924` | 0,9474 | (menunggu asisten) |
| Preview 2 (21 Okt) | tag `preview-2` | 0,9762 | (menunggu asisten) |
| Final (25 Okt)     | | | |

**Bobot model untuk asisten** (`model.pth` 111,4 MB > batas 100 MB git, jadi dilampirkan sebagai asset release — sesuai kebijakan Preview 2/Final). Direct download URL:

```
https://github.com/stcahya/DL02-klasifikasi-sampah/releases/download/preview-2/model.pth
```

Letakkan file hasil unduhan sebagai `submission/model.pth`, lalu jalankan `python evaluate.py --submission ./submission --data <data>`.

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

- **Riwayat model** (semuanya diukur dengan `evaluate.py` asli pada holdout beku yang sama, 413 gambar): ResNet-18 15 epoch **0,9373** → + hflip TTA **0,9403** → late fusion ResNet-18 + EfficientNet-B0 **0,9474** (salah 26 → 24 → 21 gambar) → **ConvNeXt-Tiny 30 epoch 0,9762** (salah 10 gambar). Di fusi lama `kertas_kardus` kelas terlemah (F1 0,9273); di ConvNeXt justru terkuat (0,9818), sisa error hanya `saset↔styrofoam` dan `gelas→botol`.
- **Angka lokal di atas tidak bisa dibandingkan langsung dengan 0,92** milik baseline starter di validation set tersembunyi (±740 gambar, split berbeda) — pembanding resminya kolom ketiga tabel di atas.
- **Sapuan gerbang fusi** (bobot cabang ResNet-18, TTA nyala): 0,25 → 0,9147 | 0,40 → 0,9295 | 0,50 → 0,9312 | 0,60 → 0,9417 | **0,75 → 0,9474**. Gerbang asimetris dipilih karena cabang ResNet-18 jauh lebih kuat (0,9373 vs 0,8698); kalau fusi tidak menolong di val tersembunyi, model turun ke level model terbaik kita, bukan jatuh.
- **Catatan kejujuran soal noise**: selisih antar-kandidat teratas hanya 1–3 gambar dari 413, jadi tidak semuanya signifikan secara statistik. Yang membuat fusi dipertahankan adalah arahnya konsisten (tren monoton terhadap gerbang) ditambah bukti bahwa fusi memang mengerjakan hal besar: dua cabang lemah (0,8666 & 0,8698) digabung 0,5/0,5 menghasilkan **0,9410** (+0,07 dari cabang terbaiknya).
- **Gagal / terpelajar**: (1) ensemble dua backbone dilatih bersama — cabang jadi lemah, jalan keluarnya late fusion; (2) `patience` dihitung per **epoch** sementara `bs 64` cuma 36 langkah/epoch (setengah run lokal) → run berhenti di epoch 8 dengan puncak di epoch 3, jadi `EPOCHS`/`patience` harus diskalakan ke jumlah langkah; (3) `--workers 4` → `MemoryError` (RAM 15,2 GB, sisa 2,5 GB; tiap worker ±400 MB) → pakai `--workers 2`, dan untuk model ensemble lokal lebih aman `--workers 0`; (4) TTA menolong model kuat (+0,003 pada ResNet-18) tapi memerosotkan ensemble yang cabangnya lemah (0,9410 → 0,8869 di percobaan pertama).
- **ConvNeXt-Tiny → submission Preview 2** (`submission/model.py` = `experiments/convnext_model.py`, dilatih lewat notebook Colab): backbone keluarga ketiga dengan **dropout 0,3 di head** dan **stochastic depth 0,1**, dilatih dengan flag optimasi baru di `train.py` — `--randaugment` (RandAugment 2 op magnitude 9 + crop lebih agresif + erasing p=0,5), `--mixup 0.2` (Mixup+CutMix, loss jadi SoftTargetCrossEntropy), dan `--warmup 3` (linear warmup 3 epoch lalu cosine decay per batch; scheduler lama `CosineAnnealingLR` dibiarkan sebagai default agar resep lama reproducible). Sendirian 0,9762 > fusi 0,9474, jadi dipromosikan langsung tanpa perlu jadi cabang fusi. Ukuran 111,4 MB → bobot didistribusikan lewat GitHub Release.
- **Berikutnya (Final)**: opsi terbuka — (1) fusi ConvNeXt + ResNet-18 (dua keluarga backbone berbeda, potensi +0,005–0,01), (2) TTA di ConvNeXt, (3) lanjut latih lebih lama. Dipilih setelah angka resmi Preview 2 keluar.

## Cara Reproduksi

```bash
pip install -r requirements.txt  # env lokal tim (torch/torchvision/timm/pillow/numpy versi terpin)
python make_split.py --check     # verifikasi holdout beku: 413 val / 2340 train, sha256 kanonik b56ef897...
python train.py                  # ResNet-18, 15 epoch (CPU ±45 menit) -> submission/model.pth + my_val/ + runs/
python evaluate.py --submission ./submission --data ./my_val
```

Reproduksi **submission yang dikirim** (ConvNeXt-Tiny, tag `preview-2`):

```bash
# 1. unduh bobot dari GitHub Release (111,4 MB)
curl -L -o submission/model.pth \
  https://github.com/stcahya/DL02-klasifikasi-sampah/releases/download/preview-2/model.pth

# 2. uji lewat grader asli
python evaluate.py --submission ./submission --data ./my_val

# alternatif latih ulang (butuh GPU; di Colab ~2 jam, hasil tidak bit-identik tapi setara):
# python train.py --model experiments/convnext_model.py --out runs/convnext_30ep.pth \
#     --epochs 30 --bs 32 --workers 2 --patience 30 --lr 2e-4 \
#     --randaugment --mixup 0.2 --warmup 3
```

Reproduksi **submission Preview 1** (late fusion, arsip): `git checkout e2a1924 -- submission/` lalu ikuti langkah fusi di history README commit tersebut.

`train.py` menyalin 413 nama dari `splits/val.txt` ke `my_val/` supaya skrip resmi `evaluate.py`
(bisa dibaca: `--data` harus berupa folder `<kelas>/*.jpg`) bisa dijalankan lokal.
Split **tidak** dibuat ulang saat pelatihan — `make_split.py --check` memastikan angkanya masih sebanding.

## Struktur Repositori

```
submission/
  model.py          # ConvNeXt-Tiny + dropout/stochastic depth; get_model() tanpa bobot
  model.pth         # TIDAK di-commit (111,4 MB > batas 100 MB git) — unduh dari GitHub Release tag preview-2
  requirements.txt  # paket TAMBAHAN untuk asisten (timm); paket dasar sudah ada di env mereka
requirements.txt    # env lokal tim (torch/torchvision/timm/numpy/pillow versi terpin)
evaluate.py         # salinan skrip evaluasi resmi (dipakai asisten)
train.py            # pelatihan pada holdout beku splits/val.txt; --model untuk melatih kandidat;
                    #   flag optimasi opsional: --randaugment, --mixup A, --warmup N
make_split.py       # bikin/periksa holdout beku 15% stratified (seed 42)
notebooks/
  train_colab.ipynb # runner Colab: clone repo, cek dataset & split, latih kandidat di GPU, unduh bobot
experiments/
  ensemble_model.py # arsitektur fusi (submission Preview 1, arsip) + resep latihnya
  effnet_model.py   # EfficientNet-B0 sendirian — cabang kedua untuk fusi
  convnext_model.py # ConvNeXt-Tiny + dropout/stochastic depth — submission Preview 2
  fuse.py           # gabungkan dua checkpoint jadi satu state_dict fusi (tanpa pelatihan)
train_baseline.py   # baseline dari asisten: split dibuat sendiri, checkpoint dipilih pakai akurasi
splits/val.txt      # 413 path relatif holdout beku — jangan diubah
my_val/             # tidak di-commit: salinan 413 gambar val untuk evaluate.py
runs/               # tidak di-commit: checkpoint tiap perbaikan + log metrik JSON
```
