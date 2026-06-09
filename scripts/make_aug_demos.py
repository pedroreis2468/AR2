"""
Demos visuais da matriz de augmentation (secção 8): uma grelha 3×2 (3 desafios ×
Base/treinado) mostrando o MESMO carro a falhar (Base) e a completar (modelo
treinado) na MESMA pista transformada. Companheiro qualitativo da matriz 2×2.

  linha 1) Espelhada            — Base ✗  vs  +Mirror ✓
  linha 2) Cones em falta       — Base ✗  vs  +Dropout ✓
  linha 3) Espelhada+sem cones  — Base ✗  vs  +Ambos ✓

Rollouts DETERMINÍSTICOS (sem domain randomization) → reprodutíveis. São exemplos
representativos; a prova estatística é a matriz da secção 8.

Saida: results/figures/aug_demos_grid.{pdf,png}
Uso:   python scripts/make_aug_demos.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.make_perception_failure_gif import rollout, _draw_scene

BASE = "runs/sac_seed0/best/best_model.zip"
MIRROR = "runs/sac_mirror_aug/best/best_model.zip"
DROP = "runs/sac_dropout_aug/best/best_model.zip"
BOTH = "runs/sac_mirror_dropout_aug/best/best_model.zip"

# (row_label, track, frac, tracks_dir, (label_esq, modelo_esq), (label_dir, modelo_dir), seed)
DEMOS = [
    ("Espelhada\n(mirrored_FSCZ24)",
     "mirrored_FSCZ24", 0.0, "tracks/mirrored", ("Base", BASE), ("+Mirror", MIRROR), 0),
    ("Cones em falta 20%\n(FSS22_V2)",
     "FSS22_V2", 0.20, "tracks", ("Base", BASE), ("+Dropout", DROP), 0),
    ("Espelhada + cones 20%\n(mirrored_FSI24)",
     "mirrored_FSI24", 0.20, "tracks/mirrored", ("Base", BASE), ("+Ambos", BOTH), 1),
]


def _panel(ax, label, model, track, frac, tracks_dir, seed, device):
    xs, ys, ths, spd, seen, td, info, ghosts = rollout(
        model, track, frac, seed=seed, mode="structural",
        tracks_dir=tracks_dir, dr=False, device=device)
    _, traj, car, head, hud = _draw_scene(ax, td, track, frac, "structural", ghosts)
    traj.set_data(xs, ys)
    car.set_data([xs[-1]], [ys[-1]])
    head.set_data([xs[-1], xs[-1] + 3.5 * np.cos(ths[-1])],
                  [ys[-1], ys[-1] + 3.5 * np.sin(ths[-1])])
    ok = info.get("laps_completed", 0) >= 1
    reason = info.get("termination_reason", "") or "fim"
    hud.set_text(f"{spd[-1]:.0f} km/h\n{len(xs)} steps")
    ax.set_title(f"{label}   {'✓ volta completa' if ok else '✗ ' + reason}",
                 color="#2ca02c" if ok else "#d62728", fontweight="bold", fontsize=11)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for r, (row_label, track, frac, tdir, left, right, seed) in enumerate(DEMOS):
        _panel(axes[r, 0], left[0], left[1], track, frac, tdir, seed, "cpu")
        _panel(axes[r, 1], right[0], right[1], track, frac, tdir, seed, "cpu")
        axes[r, 0].set_ylabel(row_label, fontweight="bold", fontsize=11)
    fig.suptitle("Augmentation ao vivo: o mesmo carro na mesma pista (Base ✗  vs  modelo treinado ✓)",
                 fontsize=14, fontweight="bold", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.path.join(ROOT, "results", "figures", "aug_demos_grid")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("[OK] aug_demos_grid.png (+ .pdf)")


if __name__ == "__main__":
    main()
