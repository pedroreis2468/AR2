"""
Tabela comparativa das ablacoes (eixo 4 do protocolo de validacao).

Le os `per_track_test.csv` de cada ablacao em results/abl_sac_*/ e compila uma
unica tabela com lap rate, cones e velocidade globais (media sobre as 5 pistas
do test split), mais o delta de lap rate face ao baseline (target entropy -0.5,
2M steps, 1 seed -- a mesma config das ablacoes, garantindo comparacao justa).

Gera:
  results/aggregated/ablations.csv
  results/aggregated/ablations.tex   (agrupada por categoria, baseline no topo)

Uso:
  python scripts/aggregate_ablations.py
"""
import argparse
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_seeds import read_per_track, tex_escape  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_SLUG = 'abl_sac_te_default'

# (slug, categoria, label legivel). Ordem = ordem de apresentacao na tabela.
ABLATIONS = [
    ('abl_sac_te_default',              'Baseline',      'Default (\\textit{target entropy} $-0.5$)'),
    ('abl_sac_no_alignment',            'Recompensa',    'Sem termo de \\textit{alignment}'),
    ('abl_sac_no_persistent_knockdown', 'Recompensa',    'Sem \\textit{persistent knockdown}'),
    ('abl_sac_no_dr',                   'Domain Rand.',  'Sem \\textit{domain randomization}'),
    ('abl_sac_no_sensor_noise',         'Sensor',        'Sem ruido de sensor'),
    ('abl_sac_fov_90',                  'Sensor',        'FOV $90^\\circ$'),
    ('abl_sac_fov_360',                 'Sensor',        'FOV $360^\\circ$'),
    ('abl_sac_range_8',                 'Sensor',        'Alcance $8$\\,m'),
    ('abl_sac_speed_60',                'Dinamica',      'Vel. max. $60$\\,km/h'),
    ('abl_sac_legacy_obs',              'Observacao',    '\\textit{Legacy obs} (18 dims)'),
    ('abl_sac_lr_low',                  'Hiperparams',   'LR $10^{-4}$'),
    ('abl_sac_buffer_1M',               'Hiperparams',   '\\textit{Buffer} $10^6$'),
    ('abl_sac_batch_512',               'Hiperparams',   '\\textit{Batch} $512$'),
    ('abl_sac_te_intermediate',         'Hiperparams',   '\\textit{Target entropy} $-1.0$'),
    ('abl_sac_te_aggressive',           'Hiperparams',   '\\textit{Target entropy} $-2.0$'),
]


def global_metrics(per_track):
    """Media (sobre pistas) de lap rate, cones e velocidade de um per_track."""
    lap = np.mean([r['lap_rate'] for r in per_track.values()])
    con = np.mean([r['cones_mean'] for r in per_track.values()])
    spd = np.mean([r['speed_mean'] for r in per_track.values()])
    return lap, con, spd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-root', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--output-dir', default=None,
                    help='Default: <results-root>/aggregated')
    args = ap.parse_args()
    out_dir = args.output_dir or os.path.join(args.results_root, 'aggregated')
    os.makedirs(out_dir, exist_ok=True)

    # Baseline primeiro (necessario para os deltas)
    base_path = os.path.join(args.results_root, BASELINE_SLUG, 'per_track_test.csv')
    if not os.path.exists(base_path):
        sys.exit(f"[ERRO] baseline ausente: {base_path}")
    base_lap, base_con, _ = global_metrics(read_per_track(base_path))

    rows = []
    for slug, cat, label in ABLATIONS:
        path = os.path.join(args.results_root, slug, 'per_track_test.csv')
        if not os.path.exists(path):
            print(f"[WARN] ausente (ignorado): {path}", file=sys.stderr)
            continue
        lap, con, spd = global_metrics(read_per_track(path))
        rows.append({
            'slug': slug, 'category': cat, 'label': label,
            'lap_rate': lap, 'cones': con, 'speed': spd,
            'delta_lap': lap - base_lap,
            'delta_cones': con - base_con,
            'is_baseline': slug == BASELINE_SLUG,
        })

    # CSV
    csv_path = os.path.join(out_dir, 'ablations.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['slug', 'category', 'lap_rate', 'cones', 'speed_kmh',
                    'delta_lap_pp', 'delta_cones'])
        for r in rows:
            w.writerow([r['slug'], r['category'], f"{r['lap_rate']:.0f}",
                        f"{r['cones']:.1f}", f"{r['speed']:.0f}",
                        '0' if r['is_baseline'] else f"{r['delta_lap']:+.0f}",
                        '0' if r['is_baseline'] else f"{r['delta_cones']:+.1f}"])

    # LaTeX (agrupada por categoria, na ordem de ABLATIONS)
    tex_path = os.path.join(out_dir, 'ablations.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("% Auto-gerada por scripts/aggregate_ablations.py\n")
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\caption{Ablacoes (SAC, 2M steps, 1 seed). Lap rate, cones e "
                "velocidade medios sobre as 5 pistas do \\textit{test split}. "
                "$\\Delta$ = variacao de lap rate face ao \\textit{baseline}.}\n")
        f.write("\\label{tab:ablations}\n")
        f.write("\\begin{tabular}{llcccc}\n\\toprule\n")
        f.write("\\textbf{Categoria} & \\textbf{Variante} & \\textbf{Voltas (\\%)} "
                "& \\textbf{$\\Delta$V (pp)} & \\textbf{Cones} "
                "& \\textbf{$\\Delta$C} \\\\\n\\midrule\n")
        last_cat = None
        for r in rows:
            cat = r['category'] if r['category'] != last_cat else ''
            last_cat = r['category']
            if cat:
                f.write("\\addlinespace[2pt]\n")
            d_lap = '---' if r['is_baseline'] else f"{r['delta_lap']:+.0f}"
            d_con = '---' if r['is_baseline'] else f"{r['delta_cones']:+.1f}"
            f.write(f"{cat} & {r['label']} & {r['lap_rate']:.0f} & {d_lap} & "
                    f"{r['cones']:.1f} & {d_con} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # Resumo no stdout, ordenado por impacto (pior -> melhor)
    print(f"\n[OK] Tabela de ablacoes em: {out_dir}")
    print(f"     Baseline (te_default) lap rate = {base_lap:.0f}%\n")
    print("     NOTA: lap rate alto + cones altos = 'completa voltas a atropelar' "
          "(eval nao termina por cones).\n")
    print(f"{'Variante':<34}{'Voltas%':>9}{'dV pp':>8}{'Cones':>8}{'dCones':>9}")
    print('-' * 68)
    for r in sorted(rows, key=lambda x: x['delta_lap']):
        if r['is_baseline']:
            dv, dc = 'base', 'base'
        else:
            dv, dc = f"{r['delta_lap']:+.0f}", f"{r['delta_cones']:+.1f}"
        print(f"{r['slug']:<34}{r['lap_rate']:>8.0f}%{dv:>8}{r['cones']:>8.1f}{dc:>9}")


if __name__ == '__main__':
    main()
