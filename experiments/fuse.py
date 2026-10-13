"""Gabungkan dua checkpoint yang dilatih terpisah jadi SATU state_dict ensemble (late fusion).

Ini bukan pelatihan: cuma menyusun ulang key state_dict supaya cocok dengan arsitektur di
experiments/ensemble_model.py, lalu memverifikasi hasilnya bisa dimuat strict=True.

    # ResNet-18 kita (key polos) + EfficientNet-B0 hasil latihan sendiri (key polos)
    python experiments/fuse.py --a submission/model.pth --b runs/b0_25ep.pth --out runs/fused.pth

    # atau mengambil satu cabang dari checkpoint ensemble yang sudah ada
    python experiments/fuse.py --a runs/ens_15ep_bs32.pth --a-prefix effnet. --b submission/model.pth --out runs/fused.pth

Gerbang (gate) rata-rata diatur di experiments/ensemble_model.py (GATE_RESNET), bukan di sini.
Setelah fusion, uji dengan grader asli:

    python -c "import pathlib,shutil; d=pathlib.Path('runs/candidate'); d.mkdir(exist_ok=True); shutil.copy('experiments/ensemble_model.py', d/'model.py'); shutil.copy('runs/fused.pth', d/'model.pth')"
    python evaluate.py --submission runs/candidate --data ./my_val --out runs/eval_fused.json
"""
import argparse
import importlib.util
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def branch_state(path, prefix, target):
    """Ambil semua tensor berawalan `prefix` dari checkpoint, ganti jadi `target`."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    ambil = {k: v for k, v in sd.items() if k.startswith(prefix)}
    if not ambil:
        raise SystemExit(f"[GAGAL] tidak ada key berawalan {prefix!r} di {path} "
                         f"(contoh key: {list(sd)[:3]})")
    return {target + k[len(prefix):]: v for k, v in ambil.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="checkpoint cabang resnet")
    ap.add_argument("--a-prefix", default="", help="awalan key di checkpoint A (mis. 'resnet.')")
    ap.add_argument("--b", required=True, help="checkpoint cabang effnet")
    ap.add_argument("--b-prefix", default="", help="awalan key di checkpoint B (mis. 'effnet.')")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("ens", ROOT / "experiments/ensemble_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    state = {}
    state.update(branch_state(a.a, a.a_prefix, "resnet."))
    state.update(branch_state(a.b, a.b_prefix, "effnet."))

    model = mod.get_model()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"[GAGAL] state hasil fusion tidak cocok: missing={missing[:5]} "
                         f"unexpected={unexpected[:5]}")
    print(f"fusion OK | GATE_RESNET={mod.GATE_RESNET} | tensor {len(state)} | "
          f"{sum(v.numel() for v in state.values())/1e6:.1f}M param")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"-> {out} ({out.stat().st_size/2**20:.1f} MB)")

    ulang = mod.get_model()
    ulang.load_state_dict(torch.load(out, map_location="cpu", weights_only=True), strict=True)
    print("verifikasi ulang strict=True: OK")


if __name__ == "__main__":
    main()
