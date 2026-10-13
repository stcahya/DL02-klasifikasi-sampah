# Studio 2 — Klasifikasi Sampah · Kelompok 12

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Stevanus Cahya Anggara | 123140038 |                  |
| 2  | Annisa Al-Qoriah | 123140030 |                  |

## Ringkasan Pendekatan

- **Arsitektur / backbone (late fusion)**: ResNet-18 + EfficientNet-B0, keduanya pretrained ImageNet dan di-fine-tune **terpisah**, lalu logits-nya digabung dengan bobot rata-rata `0,75 · ResNet-18 + 0,25 · EfficientNet-B0` (`GATE_RESNET`, konstanta — bukan parameter, jadi tidak mengubah `model.pth`). Inferensi memakai **hflip TTA** (gambar asli + cermin, logits dirata-rata). Semua digabung dalam **satu `nn.Module`** di `submission/model.py`, jadi tetap memenuhi kontrak submission (satu `model.py`, satu `model.pth`, `load_state_dict(strict=True)`).
- **Kenapa late fusion, bukan dilatih bersama**: melatih dua backbone bersamaan membuat tiap cabang belajar lebih lambat (gradiennya lewat rata-rata logits) — di percobaan 13 Okt cabang-cabangnya berakhir di 0,8666/0,8698, sedangkan ResNet-18 yang dilatih sendiri mencapai 0,9373 dengan resep identik. Penggabungan checkpoint dilakukan setelah kedua cabang selesai (`experiments/fuse.py`).
- **Strategi pelatihan**: fine-tune penuh pada holdout beku (`splits/val.txt`, 2340 train / 413 val) — AdamW `lr=3e-4`, `weight_decay=1e-4`, `CosineAnnealingLR` 15 epoch, batch 32, `CrossEntropyLoss(label_smoothing=0.1)`, augmentasi `RandomResizedCrop(224, scale=(0.6,1.0))` + `RandomHorizontalFlip` + `ColorJitter(0.3,0.3,0.3,0.05)` + `RandomErasing(p=0.25)`. Checkpoint terbaik dipilih dengan **macro-F1** memakai `metrics()` dan urutan kelas dari `evaluate.py` (metrik yang sama dengan penilaian asisten).
- **Data tambahan**: tidak ada — hanya dataset Studio 2 (2753 gambar di `train/`, 5 kelas).
- **Ukuran input & preprocessing**: `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`, identik untuk kedua backbone (default timm `efficientnet_b0` memakai mean/std yang sama), untuk validasi lokal, `my_val/`, dan grader — satu sumber kebenaran: `submission/model.py`.
- **Ukuran & waktu**: `model.pth` 58,3 MB (15,2M parameter); 413 gambar 31,7 detik di CPU → ±740 gambar ≈ 1 menit (batas asisten 15 menit).

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | `e2a1924` | 0,9474 | (menunggu asisten) |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

- **Riwayat model** (semuanya diukur dengan `evaluate.py` asli pada holdout beku yang sama, 413 gambar): ResNet-18 15 epoch **0,9373** → + hflip TTA **0,9403** → late fusion ResNet-18 + EfficientNet-B0 **0,9474** (salah 26 → 24 → 21 gambar). `kertas_kardus` tetap kelas terlemah (F1 0,9273 vs `styrofoam` 0,9510).
- **Angka lokal di atas tidak bisa dibandingkan langsung dengan 0,92** milik baseline starter di validation set tersembunyi (±740 gambar, split berbeda) — pembanding resminya kolom ketiga tabel di atas.
- **Sapuan gerbang fusi** (bobot cabang ResNet-18, TTA nyala): 0,25 → 0,9147 | 0,40 → 0,9295 | 0,50 → 0,9312 | 0,60 → 0,9417 | **0,75 → 0,9474**. Gerbang asimetris dipilih karena cabang ResNet-18 jauh lebih kuat (0,9373 vs 0,8698); kalau fusi tidak menolong di val tersembunyi, model turun ke level model terbaik kita, bukan jatuh.
- **Catatan kejujuran soal noise**: selisih antar-kandidat teratas hanya 1–3 gambar dari 413, jadi tidak semuanya signifikan secara statistik. Yang membuat fusi dipertahankan adalah arahnya konsisten (tren monoton terhadap gerbang) ditambah bukti bahwa fusi memang mengerjakan hal besar: dua cabang lemah (0,8666 & 0,8698) digabung 0,5/0,5 menghasilkan **0,9410** (+0,07 dari cabang terbaiknya).
- **Gagal / terpelajar**: (1) ensemble dua backbone dilatih bersama — cabang jadi lemah, jalan keluarnya late fusion; (2) `patience` dihitung per **epoch** sementara `bs 64` cuma 36 langkah/epoch (setengah run lokal) → run berhenti di epoch 8 dengan puncak di epoch 3, jadi `EPOCHS`/`patience` harus diskalakan ke jumlah langkah; (3) `--workers 4` → `MemoryError` (RAM 15,2 GB, sisa 2,5 GB; tiap worker ±400 MB) → pakai `--workers 2`, dan untuk model ensemble lokal lebih aman `--workers 0`; (4) TTA menolong model kuat (+0,003 pada ResNet-18) tapi memerosotkan ensemble yang cabangnya lemah (0,9410 → 0,8869 di percobaan pertama).
- **Berikutnya (Preview 2)**: latih EfficientNet-B0 sendirian dengan resep penuh (`experiments/effnet_model.py`, 25 epoch) supaya cabang keduanya sekelas ResNet-18, lalu fusi ulang — kalau cabang kedua naik ke ±0,93, fusinya diharapkan melewati 0,9474.

## Cara Reproduksi

```bash
pip install -r submission/requirements.txt
python make_split.py --check     # verifikasi holdout beku: 413 val / 2340 train, sha256 kanonik b56ef897...
python train.py                  # ResNet-18, 15 epoch (CPU ±45 menit) -> submission/model.pth + my_val/ + runs/
python evaluate.py --submission ./submission --data ./my_val
```

Reproduksi **submission yang dikirim** (late fusion):

```bash
# 1. cabang resnet: ambil bobot ResNet-18 yang sudah dipakai (commit a082efe), atau latih ulang
git show a082efe:submission/model.pth > runs/resnet18_15ep.pth
# alternatif latih ulang (±45 menit CPU, hasil tidak bit-identik tapi setara):
# python train.py --epochs 15 --out runs/resnet18_15ep.pth

# 2. cabang effnet: latih EfficientNet-B0 terpisah (butuh GPU; di Colab ~25 menit)
python train.py --model experiments/effnet_model.py --out runs/b0_25ep.pth --epochs 25 --bs 32 --patience 25

# 3. gabungkan logits kedua cabang jadi satu state_dict (tidak ada pelatihan di langkah ini)
python experiments/fuse.py --a runs/resnet18_15ep.pth --b runs/b0_25ep.pth --out runs/fused.pth

# 4. uji lewat grader asli
python -c "import pathlib,shutil; d=pathlib.Path('runs/candidate'); d.mkdir(exist_ok=True); shutil.copy('experiments/ensemble_model.py', d/'model.py'); shutil.copy('runs/fused.pth', d/'model.pth')"
python evaluate.py --submission runs/candidate --data ./my_val --out runs/eval_fused.json
```

`train.py` menyalin 413 nama dari `splits/val.txt` ke `my_val/` supaya skrip resmi `evaluate.py`
(bisa dibaca: `--data` harus berupa folder `<kelas>/*.jpg`) bisa dijalankan lokal.
Split **tidak** dibuat ulang saat pelatihan — `make_split.py --check` memastikan angkanya masih sebanding.

## Struktur Repositori

```
submission/
  model.py          # arsitektur fusi (ResNet-18 + EfficientNet-B0) + TTA; get_model() tanpa bobot
  model.pth         # state_dict fusi 58,3 MB (batas teknis 500 MB; >100 MB wajib GitHub Release/LFS)
  requirements.txt  # pin versi paket yang dipasang asisten (termasuk timm)
evaluate.py         # salinan skrip evaluasi resmi (dipakai asisten)
train.py            # pelatihan pada holdout beku splits/val.txt; --model untuk melatih kandidat
make_split.py       # bikin/periksa holdout beku 15% stratified (seed 42)
notebooks/
  train_colab.ipynb # runner Colab: clone repo, cek dataset & split, latih kandidat di GPU, unduh bobot
experiments/
  ensemble_model.py # arsitektur fusi (dipakai submission) + resep latihnya
  effnet_model.py   # EfficientNet-B0 sendirian — cabang kedua untuk fusi
  fuse.py           # gabungkan dua checkpoint jadi satu state_dict fusi (tanpa pelatihan)
train_baseline.py   # baseline dari asisten: split dibuat sendiri, checkpoint dipilih pakai akurasi
splits/val.txt      # 413 path relatif holdout beku — jangan diubah
my_val/             # tidak di-commit: salinan 413 gambar val untuk evaluate.py
runs/               # tidak di-commit: checkpoint tiap perbaikan + log metrik JSON
```
