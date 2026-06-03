"""
Gera as figuras do relatorio/notebook a partir dos CSVs agregados em
results/aggregated/. Nao re-corre avaliacao: le apenas os resumos ja
calculados por aggregate_seeds.py / aggregate_ablations.py.

Saidas (PDF + PNG) em results/figures/:
  - sac_vs_ppo.{pdf,png}        trade-off velocidade/seguranca SAC vs PPO
  - direction_bias.{pdf,png}    lap rate test vs mirror (vies direcional)
  - ablations.{pdf,png}         delta de lap rate por ablacao

As funcoes plot_* sao reutilizadas pelo notebook (notebooks/overview.ipynb)
para mostrar as mesmas figuras inline.

Uso:
  python scripts/make_report_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Nota: o backend NAO e forcado no import (para as figuras aparecerem inline
# quando o notebook importa estas funcoes). So main() forca 'Agg' (headless).

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG = os.path.join(ROOT, "results", "aggregated")
OUT = os.path.join(ROOT, "results", "figures")

# Paleta consistente (cores FS-AI / colour-blind friendly)
C_SAC = "#1f77b4"
C_PPO = "#ff7f0e"
C_MIR = "#2ca02c"


def _mean(series):
    """Converte uma coluna 'x+/-y' (ou ja numerica) na componente media x."""
    def parse(v):
        if isinstance(v, str) and "+/-" in v:
            return float(v.split("+/-")[0])
        return float(v)
    return series.map(parse)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    print(f"[OK] results/figures/{name}.pdf")


def plot_sac_vs_ppo(ax=None):
    """Barras agrupadas: lap rate (%) e cones derrubados, SAC vs PPO vs Mirror."""
    df = pd.read_csv(os.path.join(AGG, "summary_main.csv"))
    models = df["model"].tolist()
    lap = _mean(df["lap_rate"]).to_numpy()
    cones = _mean(df["cones"]).to_numpy()
    colors = [C_SAC, C_PPO, C_MIR][: len(models)]

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(models))
    ax.bar(x, lap, color=colors, alpha=0.85)
    ax.set_ylabel("Lap rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10)
    ax.set_title("Generalizacao no test split (5 pistas ineditas)")
    for xi, (lr, co) in enumerate(zip(lap, cones)):
        ax.text(xi, lr + 1.5, f"{lr:.0f}%", ha="center", fontsize=9, fontweight="bold")
        ax.text(xi, lr / 2, f"{co:.1f}\ncones", ha="center", va="center",
                fontsize=8, color="white")
    ax.grid(axis="y", alpha=0.3)
    if created:
        _save(fig, "sac_vs_ppo")
        plt.close(fig)


def plot_direction_bias(ax=None):
    """Barras agrupadas: lap rate em pistas originais vs espelhadas (held-out)."""
    df = pd.read_csv(os.path.join(AGG, "summary_direction_bias.csv"))
    models = df["model"].tolist()
    test = _mean(df["test_lap_rate"]).to_numpy()
    mirror = _mean(df["mirror_lap_rate"]).to_numpy()

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(models))
    w = 0.36
    ax.bar(x - w / 2, test, w, label="Original (test)", color="#4c78a8")
    ax.bar(x + w / 2, mirror, w, label="Espelhada (mirror)", color="#e45756")
    ax.set_ylabel("Lap rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10)
    ax.set_title("Vies direcional: original vs espelhada em Y")
    ax.legend(fontsize=8)
    for xi, (t, m) in enumerate(zip(test, mirror)):
        ax.text(xi - w / 2, t + 1.5, f"{t:.0f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, m + 1.5, f"{m:.0f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if created:
        _save(fig, "direction_bias")
        plt.close(fig)


def plot_ablations(ax=None):
    """Barras horizontais: delta de lap rate (pp) de cada ablacao vs baseline."""
    df = pd.read_csv(os.path.join(AGG, "ablations.csv"))
    df = df[df["category"] != "Baseline"].copy()
    # Ordenar por delta para leitura
    df = df.sort_values("delta_lap_pp")
    labels = df["slug"].str.replace("abl_sac_", "", regex=False)
    deltas = df["delta_lap_pp"].to_numpy()
    colors = ["#59a14f" if d >= 0 else "#e15759" for d in deltas]

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
    y = np.arange(len(labels))
    ax.barh(y, deltas, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Delta lap rate vs baseline (pp)")
    ax.set_title("Ablacoes (SAC, 2M steps) — efeito no lap rate")
    ax.axvline(0, color="black", lw=0.8)
    for yi, d in zip(y, deltas):
        ax.text(d + (0.6 if d >= 0 else -0.6), yi, f"{d:+.0f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    if created:
        _save(fig, "ablations")
        plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")  # headless: gera ficheiros sem display
    plot_sac_vs_ppo()
    plot_direction_bias()
    plot_ablations()
    print(f"\n[DONE] Figuras em {OUT}")


if __name__ == "__main__":
    main()
