"""Bikin / periksa holdout beku splits/val.txt (15% stratified, seed 42).

Split ini dipakai BERSAMA: train.py dan my_val/ harus merujuk file yang sama,
kalau tidak angka antar-anggota tidak bisa dibandingkan. Karena itu skrip ini
tidak menimpa file yang sudah ada kecuali diminta eksplisit:

    python make_split.py --check    # hitung ulang, bandingkan dengan file yang ada (aman, nol tulis)
    python make_split.py            # tulis splits/val.txt (berhenti kalau sudah ada)
    python make_split.py --force    # timpa (hanya kalau dataset-nya memang berubah)

Algoritma dipertahankan persis seperti file yang sudah dibekukan: SATU
np.random.RandomState(seed) dipakai BERURUTAN per kelas, dan `int(round(n*0.15))`.
Artinya hasilnya ikut berubah kalau urutan CLASSES, himpunan/nama file di
train/<kelas>/, atau versi numpy berubah -- jangan tambah/ganti gambar tanpa
menjalankan --check lagi.

Identitas split = isi baris (urut), bukan byte mentah: file lama memakai CRLF,
file yang ditulis skrip ini memakai LF + newline akhir.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
VAL_FRAC = 0.15
SEED = 42
DEFAULT_OUT = Path("splits/val.txt")


def build_split(data_dir, frac=VAL_FRAC, seed=SEED, classes=CLASSES):
    """-> (daftar 'kelas/nama.jpg' terurut, {kelas: (n_val, n_train)})."""
    rng = np.random.RandomState(seed)
    val, per_class = [], {}
    for c in classes:
        files = sorted(p.name for p in (data_dir / c).glob("*.jpg"))
        if not files:
            raise SystemExit(f"[GAGAL] tidak ada *.jpg di {data_dir / c}")
        shuffled = [files[i] for i in rng.permutation(len(files))]
        k = int(round(len(files) * frac))
        val += [f"{c}/{name}" for name in shuffled[:k]]
        per_class[c] = (k, len(files) - k)
    val.sort()  # urutan file: dikelompokkan per kelas, nama menaik
    return val, per_class


def canonical_hash(lines):
    """Hash isi kanonik: '\n'.join + newline akhir, tanpa peduli line-ending file."""
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train", help="folder dataset (berisi <kelas>/*.jpg)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--frac", type=float, default=VAL_FRAC)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--check", action="store_true",
                    help="jangan tulis apa pun; bandingkan hasil hitung ulang dengan file yang ada")
    ap.add_argument("--force", action="store_true", help="timpa file yang sudah ada")
    a = ap.parse_args()

    data_dir, out = Path(a.data), Path(a.out)
    val, per_class = build_split(data_dir, a.frac, a.seed)
    text = "\n".join(val) + "\n"

    print(f"data={data_dir} | val={len(val)} | train={sum(t for _, t in per_class.values())} "
          f"| seed={a.seed} frac={a.frac}")
    for c, (kv, kt) in per_class.items():
        print(f"  {c:16s} val={kv:4d} train={kt:4d} ({kv / (kv + kt):.2%})")
    print(f"sha256(kanonik hitung ulang) = {canonical_hash(val)}")

    old_lines = None
    if out.exists():
        old_bytes = out.read_bytes()
        old_lines = old_bytes.decode("utf-8").splitlines()
        print(f"sha256(kanonik {out}) = {canonical_hash(old_lines)}")
        if old_lines == val:
            note = "" if old_bytes == text.encode("utf-8") else " (isi sama; byte beda cuma line-ending/trailing newline)"
            print(f"[OK] {out} berisi split yang sama persis: reproducible, tidak ada yang ditulis.{note}")
            return
        old_set, new_set = set(old_lines), set(val)
        print(f"[BEDA] {out} beda isi. baris lama={len(old_set)} baru={len(new_set)} "
              f"hilang={len(old_set - new_set)} tambahan={len(new_set - old_set)}")
        for s in sorted(old_set - new_set)[:5]:
            print(f"  - {s}")
        for s in sorted(new_set - old_set)[:5]:
            print(f"  + {s}")
        if not (a.force and not a.check):
            raise SystemExit("[BERHENTI] split beku tidak ditimpa. Pakai --force kalau dataset memang berubah "
                             "(dan sadar: my_val/ lama harus dihapus + angka lama tidak sebanding).")
    elif a.check:
        raise SystemExit(f"[GAGAL] {out} tidak ada; jalankan tanpa --check untuk membuatnya")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(text.encode("utf-8"))  # LF eksplisit: tidak ikut translasi newline OS
    print(f"[TULIS] {out}: {len(val)} baris")


if __name__ == "__main__":
    main()