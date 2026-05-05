"""
Curvas de aprendizagem agregadas a partir de evaluations.npz dos runs SB3.

Lê os ficheiros eval/evaluations.npz de N runs (SAC com 3 seeds + PPO com 1 seed),
agrega por algoritmo e produz figuras prontas para o relatório.

Uso:
  python scripts/plot_learning_curves.py \\
      --sac-runs runs/sb3_sac_train_seed0_* runs/sb3_sac_train_seed1_* runs/sb3_sac_train_seed2_* \\
      --ppo-runs runs/sb3_ppo_train_seed0_* \\
      --output results/learning_curves.pdf
"""
import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_run(run_dir):
    """Lê eval/evaluations.npz de um run."""
    candidates = glob.glob(os.path.join(run_dir, 'eval', 'evaluations.npz'))
    if not candidates:
        # tentar como pattern
        for d in glob.glob(run_dir):
            cand = os.path.join(d, 'eval', 'evaluations.npz')
            if os.path.exists(cand):
                candidates = [cand]
                break
    if not candidates:
        raise FileNotFoundError(f"evaluations.npz não encontrado em {run_dir}")
    data = np.load(candidates[0])
    return {
        'timesteps': data['timesteps'],
        'reward_mean': data['results'].mean(axis=1),
        'reward_std':  data['results'].std(axis=1),
        'ep_length_mean': data['ep_lengths'].mean(axis=1),
    }


def align_runs(runs):
    """Trunca todos os runs ao menor número de evaluations comum."""
    min_len = min(len(r['timesteps']) for r in runs)
    return [{k: v[:min_len] for k, v in r.items()} for r in runs]


def aggregate(runs):
    """Calcula mean ± std das reward_mean ao longo dos timesteps."""
    runs = align_runs(runs)
    timesteps = runs[0]['timesteps']
    rewards = np.stack([r['reward_mean'] for r in runs])  # (n_seeds, n_evals)
    ep_lengths = np.stack([r['ep_length_mean'] for r in runs])
    return {
        'timesteps': timesteps,
        'reward_mean': rewards.mean(axis=0),
        'reward_std':  rewards.std(axis=0),
        'ep_length_mean': ep_lengths.mean(axis=0),
        'n_seeds': len(runs),
    }


def expand_globs(patterns):
    """Expande padrões shell-style mantendo ordem."""
    out = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        if matches:
            out.extend(matches)
        else:
            # talvez seja um path literal sem wildcard
            if os.path.isdir(p):
                out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sac-runs', nargs='+', default=[],
                        help='Diretórios (ou padrões) de runs SAC')
    parser.add_argument('--ppo-runs', nargs='+', default=[],
                        help='Diretórios (ou padrões) de runs PPO')
    parser.add_argument('--output', type=str, default='results/learning_curves.pdf')
    parser.add_argument('--smooth', type=int, default=1,
                        help='Window de smoothing (média móvel sobre evaluations)')
    args = parser.parse_args()

    sac_dirs = expand_globs(args.sac_runs)
    ppo_dirs = expand_globs(args.ppo_runs)
    print(f"[INFO] SAC runs ({len(sac_dirs)}): {sac_dirs}")
    print(f"[INFO] PPO runs ({len(ppo_dirs)}): {ppo_dirs}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    sac_max_step = None  # último step de treino SAC, para anotação

    def smooth(y, w):
        if w <= 1: return y
        return np.convolve(y, np.ones(w)/w, mode='valid')

    if sac_dirs:
        sac_runs = [load_run(d) for d in sac_dirs]
        sac_agg = aggregate(sac_runs)
        x = sac_agg['timesteps']
        sac_max_step = int(x[-1])
        m, s = sac_agg['reward_mean'], sac_agg['reward_std']
        if args.smooth > 1:
            x_s = x[args.smooth - 1:]
            m_s = smooth(m, args.smooth)
            s_s = smooth(s, args.smooth)
        else:
            x_s, m_s, s_s = x, m, s
        ax1.plot(x_s, m_s, label=f'SAC (n={sac_agg["n_seeds"]})', color='C0', lw=2)
        ax1.fill_between(x_s, m_s - s_s, m_s + s_s, alpha=0.25, color='C0')
        ax2.plot(x_s, smooth(sac_agg['ep_length_mean'], args.smooth) if args.smooth > 1
                 else sac_agg['ep_length_mean'], label=f'SAC (n={sac_agg["n_seeds"]})',
                 color='C0', lw=2)

    if ppo_dirs:
        ppo_runs = [load_run(d) for d in ppo_dirs]
        ppo_agg = aggregate(ppo_runs)
        x = ppo_agg['timesteps']
        m, s = ppo_agg['reward_mean'], ppo_agg['reward_std']
        if args.smooth > 1:
            x_s = x[args.smooth - 1:]
            m_s = smooth(m, args.smooth)
            s_s = smooth(s, args.smooth)
        else:
            x_s, m_s, s_s = x, m, s
        ax1.plot(x_s, m_s, label=f'PPO (n={ppo_agg["n_seeds"]})', color='C1', lw=2)
        if ppo_agg['n_seeds'] > 1:
            ax1.fill_between(x_s, m_s - s_s, m_s + s_s, alpha=0.25, color='C1')
        ax2.plot(x_s, smooth(ppo_agg['ep_length_mean'], args.smooth) if args.smooth > 1
                 else ppo_agg['ep_length_mean'], label=f'PPO (n={ppo_agg["n_seeds"]})',
                 color='C1', lw=2)

    ax1.set_xlabel('Timesteps de treino')
    ax1.set_ylabel('Reward média de avaliação')
    ax1.set_title('Curva de aprendizagem')

    # Linha tracejada vertical a marcar o fim do treino SAC
    if sac_max_step is not None and ppo_dirs:
        ppo_max = max(load_run(d)['timesteps'][-1] for d in ppo_dirs)
        if ppo_max > sac_max_step:
            for ax in (ax1, ax2):
                ax.axvline(sac_max_step, color='C0', ls='--', lw=1.2, alpha=0.7)
                ymin, ymax = ax.get_ylim()
                ax.text(sac_max_step, ymax * 0.95,
                        '  fim treino SAC',
                        rotation=0, va='top', ha='left',
                        color='C0', fontsize=8, alpha=0.8)

    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Timesteps de treino')
    ax2.set_ylabel('Comprimento médio do episódio')
    ax2.set_title('Estabilidade do agente')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    print(f"[OK] Gráfico guardado em: {args.output}")


if __name__ == '__main__':
    main()
