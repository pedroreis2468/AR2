"""
Probe de robustez de perceção — irmão do "espelho das pistas" (viés direcional).

Ideia (mesmo género do mirror probe): a política DEVIA ser robusta a perder uma
fração das deteções de cones a cada step. Num FS real a perceção (YOLO + stereo
depth) falha cones por oclusão, alcance e confusão de cor. Aqui varremos a
probabilidade de falha de deteção por cone e medimos a degradação (lap rate,
cones, velocidade) — revelando fragilidade de perceção que o test split limpo
esconde. O "fix" análogo ao mirror augmentation é treinar com dropout
aleatorizado (sensor DR mais agressivo).

Mecânica: o ConeSensor já tem `detection_prob` (ver env/cone_sensor.py). Este
script constrói o ambiente e injeta `env.cone_sensor.detection_prob = 1 - dropout`
antes de correr os episódios. O caminho de dropout só está ativo quando o ruído de
sensor está ligado, pelo que `domain_randomization` fica LIGADA (a DR de dinâmica
— massa/atrito/arrasto — é o piso de ruído constante, igual à avaliação per-track;
ver nota metodológica no README). O ponto dropout=0.05 reproduz o sensor default.

Saidas:
  results/perception_robustness/robustness.csv   — formato longo (label, dropout, ...)
  results/figures/perception_robustness.{pdf,png} — lap rate vs cones perdidos

Uso:
  python scripts/eval_perception_robustness.py \\
      --model "SAC:sac:runs/sac_seed2/best/best_model.zip" \\
      --model "PPO:ppo:runs/ppo_seed2/best/best_model.zip" \\
      --model "SAC+Mirror:sac:runs/sac_mirror_aug/best/best_model.zip" \\
      --split test --n-episodes 10 --dropouts 0.05 0.2 0.4 0.6 0.8
"""
import argparse
import csv
import os
import sys

import numpy as np

# Path do projeto (permite correr de qualquer diretório)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.racing_env import FSRacingEnv
from env.track_splits import TRAIN_TRACKS, TEST_TRACKS
from scripts.eval_per_track import aggregate  # reutiliza a sumarização canónica


def _auto_vecnormalize(model_path):
    """runs/<name>/best/best_model.zip -> runs/<name>/vecnormalize.pkl (se existir)."""
    run_dir = os.path.dirname(os.path.dirname(model_path))
    candidate = os.path.join(run_dir, 'vecnormalize.pkl')
    return candidate if os.path.exists(candidate) else None


def _load_model(algo, path, device):
    from stable_baselines3 import SAC, PPO
    Model = SAC if algo == 'sac' else PPO
    return Model.load(path, device=device)


def _load_vecnorm(vecnorm_path, tracks_dir, legacy_obs):
    if not vecnorm_path:
        return None
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    dummy = DummyVecEnv([lambda: FSRacingEnv(
        tracks_dir=tracks_dir,
        track_name=TRAIN_TRACKS[0],
        randomize_track=False,
        domain_randomization=False,
        use_orange_cones=not legacy_obs,
    )])
    vecnorm = VecNormalize.load(vecnorm_path, dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm


def _remove_cones_persistent(env, frac, rng):
    """Remove persistentemente uma fração dos cones azuis/amarelos da pista.

    A centerline (logo a reward, voltas e off-course) é calculada pelo loader a
    partir do conjunto COMPLETO e fica intacta — só muda o que o carro vê/embate.
    Seguro porque os conjuntos `knocked` arrancam vazios no reset.
    """
    td = env.track_data
    for key in ('blue_cones', 'yellow_cones'):
        c = td[key]
        n = len(c)
        if n == 0 or frac <= 0:
            continue
        n_keep = max(2, int(round(n * (1.0 - frac))))
        keep = np.sort(rng.choice(n, size=n_keep, replace=False))
        td[key] = c[keep]


def evaluate_track(model, track_name, frac, mode, args, vecnorm=None):
    """Corre n_episodes numa pista, degradando os cones segundo `mode`:

      - 'perception': `detection_prob = 1 - frac` — falha transiente por frame
        (um cone falhado volta no frame seguinte);
      - 'structural': remove persistentemente `frac` dos cones físicos no reset
        (buracos que ficam toda a volta; mesma centerline/pista verdadeira).

    domain_randomization fica LIGADA (a DR de dinâmica é o piso de ruído, como na
    avaliação per-track; no modo 'perception' também ativa o dropout do sensor).
    """
    env = FSRacingEnv(
        render_mode=None,
        randomize_track=False,
        domain_randomization=True,   # piso de ruído (e, em 'perception', o dropout)
        max_episode_steps=args.max_steps,
        tracks_dir=args.tracks_dir,
        track_name=track_name,
        use_orange_cones=not args.legacy_obs,
        terminate_on_cone=False,
        doo_cone_limit=999,
        max_laps=args.max_laps,
    )
    if mode == 'perception':
        # Injeta a probabilidade de deteção desejada (1 - fração perdida/frame)
        env.cone_sensor.detection_prob = float(np.clip(1.0 - frac, 0.0, 1.0))
    rng = np.random.default_rng(args.seed)   # remoção estrutural reproduzível

    metrics = {
        'reward': [], 'steps': [], 'progress': [], 'speed_kmh': [],
        'laps': [], 'cones': [], 'lap_completed': [], 'lap_time_s': [],
    }

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        if mode == 'structural':
            _remove_cones_persistent(env, frac, rng)
            obs = env._get_obs()     # 1ª observação já reflete os cones removidos
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            obs_pred = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
            action, _ = model.predict(obs_pred, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

        laps = info.get('laps_completed', 0)
        lap_completed = laps >= 1
        lap_time = (step * env.dt * env.action_repeat / max(laps, 1)
                    if lap_completed else float('nan'))

        metrics['reward'].append(total_reward)
        metrics['steps'].append(step)
        metrics['progress'].append(info.get('total_progress', 0.0))
        metrics['speed_kmh'].append(info.get('speed_kmh', 0.0))
        metrics['laps'].append(laps)
        metrics['cones'].append(info.get('cones_hit', 0))
        metrics['lap_completed'].append(int(lap_completed))
        metrics['lap_time_s'].append(lap_time)

    env.close()
    return aggregate(metrics)


def make_plot(rows, models, dropouts, out_path, xlabel=None, title=None):
    """Lap rate (%) vs fração de cones degradados, uma linha por modelo."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xs = [100.0 * d for d in dropouts]
    for label in models:
        ys = [next(r['lap_rate'] for r in rows
                   if r['label'] == label and r['dropout'] == d)
              for d in dropouts]
        ax.plot(xs, ys, marker='o', linewidth=2, markersize=5, label=label)

    ax.set_xlabel(xlabel or "Cones perdidos pela perceção (%)")
    ax.set_ylabel("Lap rate (%)")
    ax.set_title(title or "Robustez de perceção: degradação com dropout de cones")
    ax.set_ylim(-3, 103)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f"{out_path}.{ext}", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[OK] Figura: {out_path}.pdf (+ .png)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', action='append', dest='models', required=True,
                        metavar='LABEL:ALGO:PATH',
                        help='Repetível. Ex.: "SAC:sac:runs/sac_seed2/best/best_model.zip"')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test', 'all'])
    parser.add_argument('--tracks-list', nargs='+', default=None,
                        help='Lista explícita de pistas (ignora --split).')
    parser.add_argument('--tracks-dir', type=str, default='tracks')
    parser.add_argument('--mode', type=str, default='perception',
                        choices=['perception', 'structural'],
                        help="'perception': falha transiente por frame (1-detection_prob). "
                             "'structural': remove persistentemente os cones físicos.")
    parser.add_argument('--n-episodes', type=int, default=10)
    parser.add_argument('--dropouts', type=float, nargs='+',
                        default=[0.05, 0.2, 0.4, 0.6, 0.8],
                        help='Frações de cones degradados. perception: perda/frame '
                             '(0.05 = sensor default). structural: % de cones removidos.')
    parser.add_argument('--max-steps', type=int, default=5000)
    parser.add_argument('--max-laps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--legacy-obs', action='store_true')
    parser.add_argument('--vecnormalize', type=str, default='auto',
                        choices=['auto', 'none'],
                        help="'auto' (default): procura runs/<name>/vecnormalize.pkl. "
                             "'none': desliga.")
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str, default='results/perception_robustness')
    parser.add_argument('--fig-name', type=str, default='perception_robustness',
                        help='Nome do ficheiro da figura em results/figures/ (sem extensão).')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    # Parse das entradas LABEL:ALGO:PATH
    parsed = []
    for entry in args.models:
        try:
            label, algo, path = entry.split(':', 2)
        except ValueError:
            parser.error(f"--model mal formado (esperado LABEL:ALGO:PATH): {entry!r}")
        if algo not in ('sac', 'ppo'):
            parser.error(f"ALGO inválido em {entry!r}: {algo} (usa sac|ppo)")
        if not os.path.exists(path):
            parser.error(f"Modelo não encontrado: {path}")
        vecnorm_path = (None if args.vecnormalize == 'none'
                        else _auto_vecnormalize(path))
        parsed.append((label, algo, path, vecnorm_path))

    if args.tracks_list:
        tracks = args.tracks_list
    elif args.split == 'train':
        tracks = TRAIN_TRACKS
    elif args.split == 'test':
        tracks = TEST_TRACKS
    else:
        tracks = TRAIN_TRACKS + TEST_TRACKS

    dropouts = sorted(set(args.dropouts))
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Modelos : {[p[0] for p in parsed]}")
    print(f"[INFO] Pistas  : {tracks}")
    print(f"[INFO] Dropouts: {dropouts}  | n_episodes={args.n_episodes}")
    print(f"[INFO] Total   : {len(parsed)*len(dropouts)*len(tracks)*args.n_episodes} episódios\n")

    # Labels únicos preservando ordem (permite N seeds por label -> média)
    labels = list(dict.fromkeys(p[0] for p in parsed))

    rows = []
    for label, algo, path, vecnorm_path in parsed:
        model = _load_model(algo, path, args.device)
        vecnorm = _load_vecnorm(vecnorm_path, args.tracks_dir, args.legacy_obs)
        vn_tag = ' (+vecnorm)' if vecnorm is not None else ''
        print(f"=== {label} [{algo}] {path}{vn_tag} ===")
        print(f"{'drop%':>6} {'lap%':>6} {'cones':>7} {'speed':>7}")
        for d in dropouts:
            # Macro-média sobre pistas (igual à linha AGGREGATED do eval_per_track)
            per_track = [evaluate_track(model, tr, d, args.mode, args, vecnorm)
                         for tr in tracks]
            lap_rate = float(np.mean([a['lap_rate'] for a in per_track]))
            cones = float(np.mean([a['cones_mean'] for a in per_track]))
            speed = float(np.mean([a['speed_mean'] for a in per_track]))
            rows.append({'label': label, 'algo': algo, 'dropout': d,
                         'lap_rate': lap_rate, 'cones': cones, 'speed': speed})
            print(f"{100*d:6.0f} {lap_rate:6.1f} {cones:7.1f} {speed:7.1f}")
        print()

    # Agrega por (label, dropout): média entre seeds passadas com o mesmo label
    agg_rows = []
    for label in labels:
        algo = next(r['algo'] for r in rows if r['label'] == label)
        n_seeds = len({(p[2]) for p in parsed if p[0] == label})
        for d in dropouts:
            grp = [r for r in rows if r['label'] == label and r['dropout'] == d]
            agg_rows.append({
                'label': label, 'algo': algo, 'dropout': d, 'n_seeds': n_seeds,
                'lap_rate': float(np.mean([r['lap_rate'] for r in grp])),
                'cones': float(np.mean([r['cones'] for r in grp])),
                'speed': float(np.mean([r['speed'] for r in grp])),
            })

    csv_path = os.path.join(args.output_dir, 'robustness.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['label', 'algo', 'dropout', 'n_seeds',
                                               'lap_rate', 'cones', 'speed'])
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"[OK] CSV: {csv_path}")

    if not args.no_plot:
        if args.mode == 'structural':
            xlabel = "Cones físicos removidos da pista (%)"
            title = "Robustez a cones em falta: remoção persistente"
        else:
            xlabel = "Cones perdidos pela perceção (%)"
            title = "Robustez de perceção: dropout transiente de cones"
        make_plot(agg_rows, labels, dropouts,
                  os.path.join('results', 'figures', args.fig_name),
                  xlabel=xlabel, title=title)


if __name__ == '__main__':
    main()
