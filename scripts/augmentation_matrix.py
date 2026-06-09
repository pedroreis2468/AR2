"""
Matriz de robustez por augmentation (2 fraquezas × 4 modelos).

Cruza os dois *probes* de generalização — pistas **espelhadas** (viés direcional,
secção 6) e **cones em falta** (remoção estrutural persistente, secção 7) — nos
quatro modelos do design factorial:

    Base · +Mirror · +Dropout · +Ambos

Mostra que cada augmentation arruma (sobretudo) o seu eixo, que se compõem, e que
o baseline (test normal) não sofre. Reutiliza as funções de avaliação já existentes
(env idêntico, mesma metodologia estocástica com DR — ver nota no README).

Saidas:
  results/aug_matrix/structural.csv   — lap rate vs % cones removidos, por modelo
  results/aug_matrix/matrix.csv       — modelo × [test normal, espelhado, cones 10%]
  results/figures/aug_matrix.{pdf,png}— curva estrutural + heatmap 2×(3)

Uso:
  python scripts/augmentation_matrix.py            # 4 modelos por defeito
  python scripts/augmentation_matrix.py --n-episodes 10
"""
import argparse
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.track_splits import TEST_TRACKS
from scripts.eval_perception_robustness import (
    evaluate_track, _load_model, _load_vecnorm, _auto_vecnormalize)
from scripts.eval_per_track import evaluate_one_track, aggregate

# label -> lista de checkpoints (Base = 3 seeds; augs = 1 seed, como o mirror_aug)
MODELS = [
    ("Base",     ["runs/sac_seed0/best/best_model.zip",
                  "runs/sac_seed1/best/best_model.zip",
                  "runs/sac_seed2/best/best_model.zip"]),
    ("+Mirror",  ["runs/sac_mirror_aug/best/best_model.zip"]),
    ("+Dropout", ["runs/sac_dropout_aug/best/best_model.zip"]),
    ("+Ambos",   ["runs/sac_mirror_dropout_aug/best/best_model.zip"]),
]
MIRRORED = ["mirrored_" + t for t in TEST_TRACKS]   # test split espelhado
COL = {"Base": "tab:gray", "+Mirror": "tab:blue",
       "+Dropout": "tab:orange", "+Ambos": "tab:green"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--dropouts", type=float, nargs="+",
                    default=[0.0, 0.05, 0.10, 0.20])
    ap.add_argument("--mirror-at", type=float, default=0.10,
                    help="Nível de remoção usado na coluna 'cones em falta' da matriz.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out-dir", type=str, default="results/aug_matrix")
    args = ap.parse_args()

    dropouts = sorted(set(args.dropouts + [args.mirror_at]))
    A = SimpleNamespace(max_steps=5000, tracks_dir="tracks", legacy_obs=False,
                        max_laps=1, n_episodes=args.n_episodes, seed=args.seed,
                        no_randomization=False)
    A_mir = SimpleNamespace(**{**A.__dict__, "tracks_dir": "tracks/mirrored"})
    os.makedirs(args.out_dir, exist_ok=True)

    struct_rows, matrix = [], []
    print(f"{'modelo':<10} {'test':>6} {'espelh':>7} {'cones@'+str(int(args.mirror_at*100))+'%':>8}")
    print("-" * 36)
    for label, paths in MODELS:
        # --- estrutural: lap rate por nível de remoção (média de seeds × pistas) ---
        per_level = {d: [] for d in dropouts}
        mirror_vals = []
        for p in paths:
            if not os.path.exists(p):
                sys.exit(f"[ERRO] checkpoint em falta: {p}")
            model = _load_model("sac", p, args.device)
            vec = _load_vecnorm(_auto_vecnormalize(p), A.tracks_dir, A.legacy_obs)
            for d in dropouts:
                lap = np.mean([evaluate_track(model, t, d, "structural", A, vec)["lap_rate"]
                               for t in TEST_TRACKS])
                per_level[d].append(lap)
            # --- espelhado: lap rate nas pistas espelhadas (mesma metodologia) ---
            vec_m = _load_vecnorm(_auto_vecnormalize(p), A_mir.tracks_dir, A_mir.legacy_obs)
            lap_m = np.mean([aggregate(evaluate_one_track(model, t, A_mir.n_episodes, A_mir, vec_m))["lap_rate"]
                             for t in MIRRORED])
            mirror_vals.append(lap_m)

        for d in dropouts:
            struct_rows.append({"label": label, "dropout": d,
                                "lap_rate": float(np.mean(per_level[d]))})
        test_normal = float(np.mean(per_level[0.0]))
        cones_at = float(np.mean(per_level[args.mirror_at]))
        espelh = float(np.mean(mirror_vals))
        matrix.append({"label": label, "test_normal": test_normal,
                       "espelhado": espelh, "cones_em_falta": cones_at,
                       "n_seeds": len(paths)})
        print(f"{label:<10} {test_normal:6.0f} {espelh:7.0f} {cones_at:8.0f}")

    # --- CSVs ---
    with open(os.path.join(args.out_dir, "structural.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "dropout", "lap_rate"])
        w.writeheader(); w.writerows(struct_rows)
    with open(os.path.join(args.out_dir, "matrix.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "test_normal", "espelhado",
                                          "cones_em_falta", "n_seeds"])
        w.writeheader(); w.writerows(matrix)
    print(f"\n[OK] CSVs em {args.out_dir}/")

    make_figure(struct_rows, matrix, dropouts, args.mirror_at,
                os.path.join("results", "figures", "aug_matrix"))


def make_figure(struct_rows, matrix, dropouts, mirror_at, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [m["label"] for m in matrix]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.2),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    # painel esquerdo: curva estrutural (4 modelos)
    for label in labels:
        ys = [next(r["lap_rate"] for r in struct_rows
                   if r["label"] == label and r["dropout"] == d) for d in dropouts]
        axL.plot([100 * d for d in dropouts], ys, marker="o", lw=2, ms=5,
                 color=COL.get(label), label=label)
    axL.set_xlabel("Cones físicos removidos da pista (%)")
    axL.set_ylabel("Lap rate (%)"); axL.set_ylim(-3, 103); axL.grid(alpha=0.3)
    axL.set_title("Cones em falta — curva por augmentation")
    axL.legend(frameon=False, fontsize=9)

    # painel direito: heatmap modelo × [test normal, espelhado, cones@X%]
    cols = ["Test\n(normal)", "Espelhado", f"Cones em\nfalta {int(mirror_at*100)}%"]
    M = np.array([[m["test_normal"], m["espelhado"], m["cones_em_falta"]] for m in matrix])
    im = axR.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    axR.set_xticks(range(3)); axR.set_xticklabels(cols, fontsize=9)
    axR.set_yticks(range(len(labels))); axR.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(3):
            axR.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color="black" if 25 < M[i, j] < 80 else "white")
    axR.set_title("Matriz de robustez (lap rate %)")
    fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04, label="lap rate %")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] Figura: {out_path}.pdf (+ .png)")


if __name__ == "__main__":
    main()
