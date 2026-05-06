"""
Avaliação per-track de modelos treinados.

Corre N episódios em cada pista de um split (test, train, ou all) e gera:
  - results/<run>/per_track.csv   — métricas por pista
  - results/<run>/per_track.tex   — tabela LaTeX pronta para o relatório
  - stdout                        — resumo agregado

Uso:
  python scripts/eval_per_track.py --model runs/sb3_sac_train_seed0_*/best/best_model.zip \\
      --split test --n-episodes 10
  python scripts/eval_per_track.py --model runs/sb3_sac_train_seed0_*/best/best_model.zip \\
      --split test --n-episodes 10 --vecnormalize runs/sb3_sac_train_seed0_*/vecnormalize.pkl
"""
import argparse
import os
import sys
import csv
import numpy as np

# Path do projeto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.racing_env import FSRacingEnv
from env.track_splits import TRAIN_TRACKS, TEST_TRACKS


def evaluate_one_track(model, track_name, n_episodes, args, vecnorm=None):
    """Corre n_episodes numa pista. Retorna dict de listas com métricas.

    A pista é fixa (track_name), mas domain_randomization=True introduz
    variabilidade entre episódios via:
      - massa, atrito e arrasto do veículo (±15-20%)
      - posição inicial lateral (±0.5m)
      - velocidade inicial (0-3 m/s)
      - ruído do sensor de cones
    Cada episódio recebe uma seed distinta para reprodutibilidade.
    """
    env = FSRacingEnv(
        render_mode=None,
        randomize_track=False,
        domain_randomization=not args.no_randomization,
        max_episode_steps=args.max_steps,
        tracks_dir=args.tracks_dir,
        track_name=track_name,
        use_orange_cones=not args.legacy_obs,
        terminate_on_cone=False,
        doo_cone_limit=999,
        max_laps=args.max_laps,
    )

    metrics = {
        'reward': [], 'steps': [], 'progress': [], 'speed_kmh': [],
        'laps': [], 'cones': [], 'lap_completed': [], 'lap_time_s': [],
    }

    for ep in range(n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        while not done:
            obs_pred = obs
            if vecnorm is not None:
                obs_pred = vecnorm.normalize_obs(obs)
            action, _ = model.predict(obs_pred, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

        laps = info.get('laps_completed', 0)
        lap_completed = laps >= 1
        # Lap time = steps × dt × action_repeat / laps (apenas se completou)
        if lap_completed:
            lap_time = step * env.dt * env.action_repeat / max(laps, 1)
        else:
            lap_time = float('nan')

        metrics['reward'].append(total_reward)
        metrics['steps'].append(step)
        metrics['progress'].append(info.get('total_progress', 0.0))
        metrics['speed_kmh'].append(info.get('speed_kmh', 0.0))
        metrics['laps'].append(laps)
        metrics['cones'].append(info.get('cones_hit', 0))
        metrics['lap_completed'].append(int(lap_completed))
        metrics['lap_time_s'].append(lap_time)

    env.close()
    return metrics


def aggregate(metrics):
    """Sumariza um dict de listas de métricas."""
    arr = lambda k: np.asarray(metrics[k], dtype=float)
    out = {
        'lap_rate':    100.0 * arr('lap_completed').mean(),  # %
        'cones_mean':  arr('cones').mean(),
        'cones_std':   arr('cones').std(),
        'speed_mean':  arr('speed_kmh').mean(),
        'speed_std':   arr('speed_kmh').std(),
        'reward_mean': arr('reward').mean(),
        'reward_std':  arr('reward').std(),
    }
    # Lap time só sobre as voltas completadas
    lap_times = arr('lap_time_s')
    completed = lap_times[~np.isnan(lap_times)]
    if len(completed) > 0:
        out['laptime_mean'] = completed.mean()
        out['laptime_std']  = completed.std()
    else:
        out['laptime_mean'] = float('nan')
        out['laptime_std']  = float('nan')
    return out


def write_latex_table(rows, path, split_name):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("% Tabela auto-gerada por scripts/eval_per_track.py\n")
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write(f"\\caption{{Desempenho por pista no split \\textit{{{split_name}}} "
                f"(N=10 episódios, política determinística).}}\n")
        f.write("\\label{tab:per_track_" + split_name + "}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("\\textbf{Pista} & \\textbf{Voltas (\\%)} & \\textbf{Cones} "
                "& \\textbf{Vel.\\,m\\'edia (km/h)} & \\textbf{Lap time (s)} \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            laptime = (f"{r['laptime_mean']:.1f} $\\pm$ {r['laptime_std']:.1f}"
                       if not np.isnan(r['laptime_mean']) else "---")
            f.write(f"{r['track']} & {r['lap_rate']:.0f} & "
                    f"{r['cones_mean']:.1f} $\\pm$ {r['cones_std']:.1f} & "
                    f"{r['speed_mean']:.1f} $\\pm$ {r['speed_std']:.1f} & "
                    f"{laptime} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--algo', type=str, default='sac', choices=['sac', 'ppo'])
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test', 'all'])
    parser.add_argument('--n-episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=5000)
    parser.add_argument('--max-laps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tracks-dir', type=str, default='tracks')
    parser.add_argument('--tracks-list', nargs='+', default=None,
                        help='Lista explicita de nomes de pistas (sem .yaml). '
                             'Se dado, ignora --split. Util para avaliar em pistas '
                             'fora dos splits originais (e.g. mirrored, extreme).')
    parser.add_argument('--legacy-obs', action='store_true')
    parser.add_argument('--no-randomization', action='store_true',
                        help='Desligar domain randomization na avaliação (por defeito está LIGADA '
                             'para introduzir variabilidade entre os N episódios por pista).')
    parser.add_argument('--vecnormalize', type=str, default=None,
                        help='Caminho para vecnormalize.pkl (se foi usado no treino)')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Default: results/<basename do --model>')
    args = parser.parse_args()

    from stable_baselines3 import SAC, PPO
    Model = SAC if args.algo == 'sac' else PPO
    model = Model.load(args.model, device=args.device)

    vecnorm = None
    if args.vecnormalize and os.path.exists(args.vecnormalize):
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        # Dummy env só para carregar o vecnorm
        dummy = DummyVecEnv([lambda: FSRacingEnv(
            tracks_dir=args.tracks_dir,
            track_name=TRAIN_TRACKS[0],
            randomize_track=False,
            domain_randomization=False,
            use_orange_cones=not args.legacy_obs,
        )])
        vecnorm = VecNormalize.load(args.vecnormalize, dummy)
        vecnorm.training = False
        vecnorm.norm_reward = False

    if args.tracks_list:
        # Lista explícita: ignora o split, usa as pistas dadas
        tracks = args.tracks_list
    elif args.split == 'train':
        tracks = TRAIN_TRACKS
    elif args.split == 'test':
        tracks = TEST_TRACKS
    else:
        tracks = TRAIN_TRACKS + TEST_TRACKS

    out_dir = args.output_dir or os.path.join(
        'results',
        os.path.splitext(os.path.basename(args.model))[0]
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Resultados em: {out_dir}")
    print(f"[INFO] Split={args.split}, pistas={tracks}, n_episodes={args.n_episodes}\n")

    rows = []
    print(f"{'Pista':<12} {'Lap%':>6} {'Cones':>14} {'Speed':>14} {'Laptime':>14}")
    print('-' * 64)
    for tr in tracks:
        m = evaluate_one_track(model, tr, args.n_episodes, args, vecnorm=vecnorm)
        agg = aggregate(m)
        agg['track'] = tr
        rows.append(agg)
        laptime_str = (f"{agg['laptime_mean']:.1f}±{agg['laptime_std']:.1f}"
                       if not np.isnan(agg['laptime_mean']) else "---")
        print(f"{tr:<12} {agg['lap_rate']:5.0f}% "
              f"{agg['cones_mean']:5.1f}±{agg['cones_std']:.1f}     "
              f"{agg['speed_mean']:5.1f}±{agg['speed_std']:.1f}     "
              f"{laptime_str:>12}")

    # CSV
    csv_path = os.path.join(out_dir, f'per_track_{args.split}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # LaTeX
    tex_path = os.path.join(out_dir, f'per_track_{args.split}.tex')
    write_latex_table(rows, tex_path, args.split)

    # Agregado
    print('-' * 64)
    arr = lambda k: np.array([r[k] for r in rows])
    print(f"{'AGGREGATED':<12} {arr('lap_rate').mean():5.0f}% "
          f"{arr('cones_mean').mean():5.1f}     "
          f"{arr('speed_mean').mean():5.1f}     "
          f"{np.nanmean(arr('laptime_mean')):.1f}")
    print(f"\n[OK] CSV: {csv_path}")
    print(f"[OK] LaTeX: {tex_path}")


if __name__ == '__main__':
    main()
