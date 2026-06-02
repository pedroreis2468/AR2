"""
Agregacao multi-seed dos resultados per-track.

Le os CSVs `per_track_*.csv` produzidos por scripts/eval_per_track.py para
varias seeds e gera as tabelas agregadas que o relatorio precisa, reproduzindo
automaticamente os numeros que ate aqui eram calculados a mao:

  results/aggregated/
    summary_main.{csv,tex}            Resumo SAC vs PPO vs Mirror (test): lap%, cones, vel.
    summary_direction_bias.{csv,tex}  Test vs Mirror (lap rate) + delta, por modelo
    per_track_SAC_test.{csv,tex}      x-bar +/- sigma ENTRE seeds, por pista (SAC)
    per_track_PPO_test.{csv,tex}      idem (PPO)

Convencao estatistica: o agregado global de um modelo (ex.: lap rate "66%") e a
media, ENTRE seeds, do valor global de cada seed (= media sobre as pistas dessa
seed). O +/- reportado e o desvio-padrao ENTRE seeds -> mede a consistencia
seed-a-seed. Nas tabelas per-track, o +/- e o desvio ENTRE seeds para essa pista.

Uso:
  python scripts/aggregate_seeds.py
  python scripts/aggregate_seeds.py --results-root results --output-dir results/aggregated
"""
import argparse
import csv
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Familias do projeto: label -> diretorios de resultados (uma seed por diretorio).
FAMILIES_TEST = {
    'SAC':             ['sac_seed0', 'sac_seed1', 'sac_seed2'],
    'PPO':             ['ppo_seed0', 'ppo_seed1', 'ppo_seed2'],
    'SAC+Mirror Aug.': ['sac_mirror_aug'],
}
FAMILIES_MIRROR = {
    'SAC':             ['sac_seed0_mirrored', 'sac_seed1_mirrored', 'sac_seed2_mirrored'],
    'PPO':             ['ppo_seed0_mirrored', 'ppo_seed1_mirrored', 'ppo_seed2_mirrored'],
    'SAC+Mirror Aug.': ['sac_mirror_aug_mirrored'],
}
CSV_TEST = 'per_track_test.csv'


def read_per_track(path):
    """Le um per_track CSV -> dict {track: {coluna: float}} (track fica string)."""
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out[row['track']] = {
                k: (v if k == 'track' else float(v)) for k, v in row.items()
            }
    return out


def load_family(run_dirs, csv_name, results_root):
    """Le o per_track de cada seed existente. Retorna lista de dicts {track: row}."""
    seeds = []
    for rd in run_dirs:
        path = os.path.join(results_root, rd, csv_name)
        if os.path.exists(path):
            seeds.append(read_per_track(path))
        else:
            print(f"[WARN] ausente (ignorado): {path}", file=sys.stderr)
    return seeds


def global_per_seed(seeds, col):
    """Valor global de `col` por seed = nanmean sobre as pistas dessa seed.

    nanmean ignora pistas sem volta completa (laptime = nan), evitando que
    uma pista falhada contamine a media de lap time."""
    vals = []
    for s in seeds:
        track_vals = np.array([s[tr][col] for tr in s], dtype=float)
        vals.append(np.nanmean(track_vals))
    return np.array(vals, dtype=float)


def mean_std_across_seeds(seeds, col):
    """(media, desvio, n_seeds) do valor global de `col` entre seeds."""
    g = global_per_seed(seeds, col)
    g = g[~np.isnan(g)]
    if len(g) == 0:
        return float('nan'), float('nan'), 0
    return float(g.mean()), float(g.std()), len(g)


def per_track_across_seeds(seeds, col):
    """{track: (media, desvio, n)} do valor de `col` por pista, entre seeds."""
    out = {}
    for tr in seeds[0]:
        vals = np.array([s[tr][col] for s in seeds if tr in s], dtype=float)
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            out[tr] = (float('nan'), float('nan'), 0)
        else:
            out[tr] = (float(valid.mean()), float(valid.std()), len(valid))
    return out


def tex_escape(s):
    """Escapa caracteres especiais de LaTeX em nomes de pista (ex.: FSS22_V2)."""
    return s.replace('_', r'\_')


def fmt(mean, std, n, decimals=1):
    """'x +/- s' (LaTeX) ou so 'x' quando ha 1 seed; '---' se nan."""
    if np.isnan(mean):
        return "---"
    if n <= 1:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def fmt_txt(mean, std, n, decimals=1):
    """Versao texto (stdout/CSV) de fmt."""
    if np.isnan(mean):
        return "---"
    if n <= 1:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}+/-{std:.{decimals}f}"


# --------------------------------------------------------------------------- #
# Tabela 1: resumo principal (lap rate, cones, velocidade) no test split
# --------------------------------------------------------------------------- #
def build_summary_main(families, results_root, out_dir):
    rows = []
    for label, run_dirs in families.items():
        seeds = load_family(run_dirs, CSV_TEST, results_root)
        if not seeds:
            continue
        lap = mean_std_across_seeds(seeds, 'lap_rate')
        con = mean_std_across_seeds(seeds, 'cones_mean')
        spd = mean_std_across_seeds(seeds, 'speed_mean')
        rows.append({
            'model': label, 'n_seeds': lap[2],
            'lap_rate': lap, 'cones': con, 'speed': spd,
        })

    # CSV
    csv_path = os.path.join(out_dir, 'summary_main.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'n_seeds', 'lap_rate', 'cones', 'speed_kmh'])
        for r in rows:
            w.writerow([
                r['model'], r['n_seeds'],
                fmt_txt(*r['lap_rate'], decimals=0),
                fmt_txt(*r['cones'], decimals=1),
                fmt_txt(*r['speed'], decimals=0),
            ])

    # LaTeX
    tex_path = os.path.join(out_dir, 'summary_main.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("% Auto-gerada por scripts/aggregate_seeds.py\n")
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\caption{Resumo no \\textit{test split} FS-AI (5 pistas ineditas, "
                "10 episodios/pista, politica deterministica). Valores: "
                "$\\bar{x}\\pm\\sigma$ entre seeds.}\n")
        f.write("\\label{tab:summary_main}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("\\textbf{Modelo} & \\textbf{Seeds} & \\textbf{Voltas (\\%)} "
                "& \\textbf{Cones} & \\textbf{Vel. (km/h)} \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['model']} & {r['n_seeds']} & "
                    f"{fmt(*r['lap_rate'], decimals=0)} & "
                    f"{fmt(*r['cones'], decimals=1)} & "
                    f"{fmt(*r['speed'], decimals=0)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return rows, csv_path, tex_path


# --------------------------------------------------------------------------- #
# Tabela 2: vies direcional (test vs mirrored)
# --------------------------------------------------------------------------- #
def build_direction_bias(fam_test, fam_mirror, results_root, out_dir):
    rows = []
    for label in fam_test:
        st = load_family(fam_test[label], CSV_TEST, results_root)
        sm = load_family(fam_mirror.get(label, []), CSV_TEST, results_root)
        if not st or not sm:
            continue
        lt = mean_std_across_seeds(st, 'lap_rate')
        lm = mean_std_across_seeds(sm, 'lap_rate')
        delta = lm[0] - lt[0]
        rows.append({'model': label, 'test': lt, 'mirror': lm, 'delta': delta})

    csv_path = os.path.join(out_dir, 'summary_direction_bias.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'test_lap_rate', 'mirror_lap_rate', 'delta_pp'])
        for r in rows:
            w.writerow([r['model'], fmt_txt(*r['test'], decimals=0),
                        fmt_txt(*r['mirror'], decimals=0), f"{r['delta']:+.0f}"])

    tex_path = os.path.join(out_dir, 'summary_direction_bias.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("% Auto-gerada por scripts/aggregate_seeds.py\n")
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\caption{Vies direcional: lap rate em pistas originais "
                "(\\textit{test}) vs espelhadas em Y (\\textit{mirror}, held-out). "
                "$\\bar{x}\\pm\\sigma$ entre seeds.}\n")
        f.write("\\label{tab:direction_bias}\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("\\textbf{Modelo} & \\textbf{Test (\\%)} & \\textbf{Mirror (\\%)} "
                "& \\textbf{$\\Delta$ (pp)} \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['model']} & {fmt(*r['test'], decimals=0)} & "
                    f"{fmt(*r['mirror'], decimals=0)} & {r['delta']:+.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return rows, csv_path, tex_path


# --------------------------------------------------------------------------- #
# Tabela 3: per-track com x-bar +/- sigma entre seeds
# --------------------------------------------------------------------------- #
def build_per_track(label, run_dirs, results_root, out_dir):
    seeds = load_family(run_dirs, CSV_TEST, results_root)
    if len(seeds) < 2:
        return None  # per-track entre seeds so faz sentido com >=2 seeds
    lap = per_track_across_seeds(seeds, 'lap_rate')
    con = per_track_across_seeds(seeds, 'cones_mean')
    spd = per_track_across_seeds(seeds, 'speed_mean')
    lpt = per_track_across_seeds(seeds, 'laptime_mean')
    tracks = list(seeds[0].keys())

    csv_path = os.path.join(out_dir, f'per_track_{label}_test.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['track', 'lap_rate', 'cones', 'speed_kmh', 'laptime_s', 'n_seeds'])
        for tr in tracks:
            w.writerow([tr, fmt_txt(*lap[tr], 0), fmt_txt(*con[tr], 1),
                        fmt_txt(*spd[tr], 1), fmt_txt(*lpt[tr], 1), lap[tr][2]])

    tex_path = os.path.join(out_dir, f'per_track_{label}_test.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("% Auto-gerada por scripts/aggregate_seeds.py\n")
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write(f"\\caption{{Desempenho per-track do {label} no \\textit{{test split}} "
                f"({len(seeds)} seeds, 10 episodios/pista). $\\bar{{x}}\\pm\\sigma$ "
                f"entre seeds.}}\n")
        f.write("\\label{tab:per_track_" + label + "}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("\\textbf{Pista} & \\textbf{Voltas (\\%)} & \\textbf{Cones} "
                "& \\textbf{Vel. (km/h)} & \\textbf{Lap time (s)} \\\\\n\\midrule\n")
        for tr in tracks:
            f.write(f"{tex_escape(tr)} & {fmt(*lap[tr], 0)} & {fmt(*con[tr], 1)} & "
                    f"{fmt(*spd[tr], 1)} & {fmt(*lpt[tr], 1)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return csv_path, tex_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-root', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--output-dir', default=None,
                    help='Default: <results-root>/aggregated')
    args = ap.parse_args()

    out_dir = args.output_dir or os.path.join(args.results_root, 'aggregated')
    os.makedirs(out_dir, exist_ok=True)

    main_rows, _, _ = build_summary_main(FAMILIES_TEST, args.results_root, out_dir)
    bias_rows, _, _ = build_direction_bias(FAMILIES_TEST, FAMILIES_MIRROR,
                                            args.results_root, out_dir)
    for label, run_dirs in FAMILIES_TEST.items():
        build_per_track(label, run_dirs, args.results_root, out_dir)

    # Resumo no stdout
    print(f"\n[OK] Tabelas agregadas em: {out_dir}\n")
    print("=== Resumo principal (test split, x-bar +/- sigma entre seeds) ===")
    print(f"{'Modelo':<18}{'Seeds':>6}{'Voltas%':>14}{'Cones':>14}{'Vel km/h':>12}")
    for r in main_rows:
        print(f"{r['model']:<18}{r['n_seeds']:>6}"
              f"{fmt_txt(*r['lap_rate'],0):>14}{fmt_txt(*r['cones'],1):>14}"
              f"{fmt_txt(*r['speed'],0):>12}")
    print("\n=== Vies direcional (lap rate test vs mirror) ===")
    print(f"{'Modelo':<18}{'Test%':>14}{'Mirror%':>14}{'Delta pp':>10}")
    for r in bias_rows:
        print(f"{r['model']:<18}{fmt_txt(*r['test'],0):>14}"
              f"{fmt_txt(*r['mirror'],0):>14}{r['delta']:>+10.0f}")


if __name__ == '__main__':
    main()
