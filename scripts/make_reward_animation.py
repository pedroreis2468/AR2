"""
Gera uma animacao (GIF) de um modelo treinado a conduzir uma volta, com a
EVOLUCAO DA RECOMPENSA em sincronia: painel esquerdo = carro na pista;
painel direito = componentes da reward a crescer passo-a-passo.

Os dados sao REAIS (rollout deterministico de um modelo SB3); nada e inventado.
As formulas dos componentes sao identicas a env/racing_env._compute_reward.

Saida: results/figures/reward_evolution.gif

Uso:
  python scripts/make_reward_animation.py
  python scripts/make_reward_animation.py --track FSI24 --model runs/sac_seed0/best/best_model.zip
"""
import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

COLORS = {"r_progress": "#2ca02c", "r_alignment": "#1f77b4", "r_smooth": "#9467bd",
          "r_lateral": "#ff7f0e", "r_time": "#7f7f7f"}


def rollout(model_path, track, max_steps=3000, device="cpu"):
    """Corre uma volta e devolve trajetoria + componentes da reward por step."""
    from stable_baselines3 import SAC
    from env.racing_env import FSRacingEnv
    model = SAC.load(os.path.join(ROOT, model_path), device=device)
    env = FSRacingEnv(render_mode=None, randomize_track=False, domain_randomization=False,
                      max_episode_steps=max_steps, tracks_dir=os.path.join(ROOT, "tracks"),
                      track_name=track, use_orange_cones=True, terminate_on_cone=False,
                      doo_cone_limit=999, max_laps=1)
    obs, info = env.reset(seed=0)
    vmax = env.vehicle_params.max_speed
    prev_prog, prev_steer = 0.0, 0.0
    xs, ys, ths, spd = [], [], [], []
    comps = {k: [] for k in COLORS}
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        cl = env.track_data["centerline"]
        d2 = np.sum((cl - np.array([env.car.x, env.car.y]))**2, axis=1)
        cl_idx = int(np.argmin(d2))
        lateral = float(np.sqrt(d2[cl_idx]))
        hw = env._get_effective_hw()
        hdg = env._get_heading_error(cl_idx)
        raw = info["total_progress"] - prev_prog; prev_prog = info["total_progress"]
        prog = max(0.0, raw)  # ignora o reset de total_progress ao cruzar a meta
        steer = float(a[0]); ratio = lateral / (hw + 1e-6)
        comps["r_progress"].append(prog * 2.0)
        comps["r_alignment"].append((env.car.v / (vmax + 1e-6)) * np.cos(hdg) * 0.4)
        comps["r_smooth"].append(-max(0.0, abs(steer - prev_steer) - 0.1) * 0.05)
        comps["r_lateral"].append(-ratio * 0.08 if ratio <= 1.0 else -0.08 - (ratio - 1.0)**2 * 2.0)
        comps["r_time"].append(-0.005)
        xs.append(env.car.x); ys.append(env.car.y); ths.append(env.car.theta)
        spd.append(info["speed_kmh"]); prev_steer = steer
    track_data = env.track_data
    env.close()
    comps = {k: np.asarray(v) for k, v in comps.items()}
    return (np.asarray(xs), np.asarray(ys), np.asarray(ths), np.asarray(spd),
            comps, track_data, info)


def build_animation(model_path="runs/sac_seed0/best/best_model.zip",
                    track="FSCZ24", stride=3, fps=12, device="cpu"):
    """Constroi e devolve (FuncAnimation, info). Nao grava."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    xs, ys, ths, spd, comps, td, info = rollout(model_path, track, device=device)
    N = len(xs)
    steps = np.arange(N)
    per_step_total = sum(comps.values())
    cum_total = np.cumsum(per_step_total)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1.0]})

    # --- painel esquerdo: pista (estatico) ---
    cl = td["centerline"]
    axL.plot(cl[:, 0], cl[:, 1], "--", color="0.7", lw=0.8, zorder=1)
    for key, c in [("blue_cones", "tab:blue"), ("yellow_cones", "gold"), ("orange_cones", "darkorange")]:
        p = td.get(key, np.zeros((0, 2)))
        if len(p):
            axL.scatter(p[:, 0], p[:, 1], s=10, color=c, zorder=2)
    axL.set_aspect("equal")
    mx, my = cl[:, 0], cl[:, 1]
    pad = 6
    axL.set_xlim(mx.min() - pad, mx.max() + pad)
    axL.set_ylim(my.min() - pad, my.max() + pad)
    axL.set_title(f"Volta em {track}")
    axL.set_xticks([]); axL.set_yticks([])
    (traj_line,) = axL.plot([], [], "-", color="crimson", lw=1.6, zorder=3)
    (car_pt,) = axL.plot([], [], "o", color="black", ms=7, zorder=5)
    (head_ln,) = axL.plot([], [], "-", color="black", lw=2, zorder=4)
    info_txt = axL.text(0.02, 0.98, "", transform=axL.transAxes, va="top", ha="left",
                        fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # --- painel direito: evolucao da reward ---
    comp_lines = {}
    for k in COLORS:
        (ln,) = axR.plot([], [], color=COLORS[k], lw=1.4, label=k)
        comp_lines[k] = ln
    cursor = axR.axvline(0, color="0.4", lw=0.8, ls=":")
    axR.axhline(0, color="black", lw=0.6)
    axR.set_xlim(0, N)
    ymin = min(0.0, min(c.min() for c in comps.values())) - 0.1
    ymax = max(c.max() for c in comps.values()) + 0.3
    axR.set_ylim(ymin, ymax)
    axR.set_xlabel("step da volta"); axR.set_ylabel("recompensa / step")
    axR.set_title("Evolução da recompensa")
    axR.legend(fontsize=8, ncol=2, loc="upper right")
    total_txt = axR.text(0.02, 0.97, "", transform=axR.transAxes, va="top", ha="left",
                         fontsize=10, fontweight="bold",
                         bbox=dict(boxstyle="round", fc="#fffbe6", alpha=0.9))
    fig.tight_layout()

    frames = list(range(0, N, stride)) + [N - 1]

    def update(i):
        traj_line.set_data(xs[:i + 1], ys[:i + 1])
        car_pt.set_data([xs[i]], [ys[i]])
        L = 3.5
        head_ln.set_data([xs[i], xs[i] + L * np.cos(ths[i])],
                         [ys[i], ys[i] + L * np.sin(ths[i])])
        info_txt.set_text(f"step {i}\n{spd[i]:.0f} km/h\nΣreward {cum_total[i]:+.0f}")
        for k, ln in comp_lines.items():
            ln.set_data(steps[:i + 1], comps[k][:i + 1])
        cursor.set_xdata([i, i])
        total_txt.set_text(f"Σreward = {cum_total[i]:+.0f}")
        return (traj_line, car_pt, head_ln, info_txt, cursor, total_txt, *comp_lines.values())

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    return anim, fig, info, fps


def make_gif(out_path=None, **kwargs):
    """Gera e grava o GIF. Devolve o caminho.

    Nao muda o backend do matplotlib (para ser seguro chamar a partir do
    notebook com %matplotlib inline). O modo script forca 'Agg' no main().
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    fps = kwargs.pop("fps", 12)
    anim, fig, info, _ = build_animation(fps=fps, **kwargs)
    if out_path is None:
        out_path = os.path.join(ROOT, "results", "figures", "reward_evolution.gif")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"[OK] {out_path}  (laps={info['laps_completed']}, cones={info['cones_hit']})")
    return out_path


def main():
    import matplotlib
    matplotlib.use("Agg")  # headless: gera o GIF sem display
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/sac_seed0/best/best_model.zip")
    ap.add_argument("--track", default="FSCZ24")
    ap.add_argument("--stride", type=int, default=3, help="1 em cada N steps vira frame")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    make_gif(out_path=args.out, model_path=args.model, track=args.track,
             stride=args.stride, fps=args.fps, device=args.device)


if __name__ == "__main__":
    main()
