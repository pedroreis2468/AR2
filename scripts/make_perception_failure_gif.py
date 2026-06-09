"""
GIF do carro a FALHAR por falta de cones — versão visual do probe de robustez.
Suporta dois modos (ver scripts/eval_perception_robustness.py):

  --mode perception : falha transiente de deteção (1 - detection_prob) por frame.
      Mostra todos os cones reais a cinza e, vivos, os que o sensor perceciona
      nesse frame (em alcance + FOV, sobreviventes ao dropout, N mais próximos).

  --mode structural : remoção PERSISTENTE de uma fração dos cones físicos.
      Mostra os cones removidos como "fantasmas" (× cinza) e os restantes sólidos
      — vê-se o carro a cair nos buracos deixados pelos cones em falta.

A trajetória é sempre REAL (a degradação atua dentro de env.step / no reset). A
centerline (logo a reward, voltas e off-course) fica intacta — só muda o que o
carro vê/embate.

Uso:
  python scripts/make_perception_failure_gif.py --mode structural --track FSI24 \\
      --dropout 0.2 --model runs/sac_seed2/best/best_model.zip --contrast
"""
import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CONE_COLORS = [('blue_cones', 'tab:blue'), ('yellow_cones', 'gold'),
               ('orange_cones', 'darkorange')]


def perceived_idx(cx, cy, cth, cones, max_range, fov, n_closest, det_prob, rng):
    """Índices dos cones percecionados — replica ConeSensor._detect_cones."""
    if len(cones) == 0:
        return np.empty(0, dtype=int)
    dx = cones[:, 0] - cx
    dy = cones[:, 1] - cy
    cos_t, sin_t = np.cos(-cth), np.sin(-cth)
    xc = dx * cos_t - dy * sin_t
    yc = dx * sin_t + dy * cos_t
    dist = np.hypot(xc, yc)
    bearing = np.arctan2(yc, xc)
    visible = (dist <= max_range) & (np.abs(bearing) <= fov / 2)
    visible &= rng.random(len(cones)) < det_prob
    if not visible.any():
        return np.empty(0, dtype=int)
    order = np.argsort(np.where(visible, dist, np.inf))[:n_closest]
    return np.array([i for i in order if visible[i]], dtype=int)


def rollout(model_path, track, frac, seed=0, mode='perception',
            max_steps=3000, device="cpu", tracks_dir="tracks", dr=True):
    """Conduz uma volta degradando os cones segundo `mode`. Devolve trajetória,
    cones percecionados/frame, track_data final, info e fantasmas (removidos).

    `tracks_dir` permite pistas espelhadas (tracks/mirrored); `dr=False` corre
    determinístico (sem domain randomization), para demos reprodutíveis.
    """
    from stable_baselines3 import SAC
    from env.racing_env import FSRacingEnv

    np.random.seed(seed)                       # rollout reproduzível (DR usa np global)
    rng = np.random.default_rng(seed)
    model = SAC.load(os.path.join(ROOT, model_path), device=device)
    env = FSRacingEnv(
        render_mode=None, randomize_track=False, domain_randomization=dr,
        max_episode_steps=max_steps, tracks_dir=os.path.join(ROOT, tracks_dir),
        track_name=track, use_orange_cones=True, terminate_on_cone=False,
        doo_cone_limit=999, max_laps=1,
    )
    if mode == 'perception':
        env.cone_sensor.detection_prob = float(np.clip(1.0 - frac, 0.0, 1.0))
    s = env.cone_sensor
    obs, info = env.reset(seed=seed)

    ghosts = {k: np.zeros((0, 2)) for k, _ in CONE_COLORS}
    if mode == 'structural':
        td = env.track_data
        for key, _ in CONE_COLORS:
            if key == 'orange_cones':
                continue                       # manter start/meta
            c = td[key]
            n = len(c)
            if n == 0 or frac <= 0:
                continue
            n_keep = max(2, int(round(n * (1.0 - frac))))
            keep = np.sort(rng.choice(n, size=n_keep, replace=False))
            mask = np.ones(n, dtype=bool)
            mask[keep] = False
            ghosts[key] = c[mask].copy()       # removidos (fantasmas)
            td[key] = c[keep]                  # restantes (pista esparsa)
        obs = env._get_obs()                   # 1ª obs reflete a remoção

    xs, ys, ths, spd, seen = [], [], [], [], []
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        cx, cy, cth = env.car.x, env.car.y, env.car.theta
        per = {}
        for key, _ in CONE_COLORS:
            cones = env.track_data.get(key, np.zeros((0, 2)))
            idx = perceived_idx(cx, cy, cth, cones, s.max_range, s.fov,
                                s.n_closest, s.detection_prob, rng)
            per[key] = cones[idx] if len(idx) else np.zeros((0, 2))
        xs.append(cx); ys.append(cy); ths.append(cth)
        spd.append(info["speed_kmh"]); seen.append(per)

    td = env.track_data
    env.close()
    return (np.array(xs), np.array(ys), np.array(ths), np.array(spd),
            seen, td, info, ghosts)


def _draw_scene(ax, td, track, frac, mode, ghosts):
    """Desenha a cena estática. Devolve handles dinâmicos (e seen_sc se aplicável)."""
    cl = td["centerline"]
    ax.plot(cl[:, 0], cl[:, 1], "--", color="0.82", lw=0.8, zorder=1)
    seen_sc = {}
    if mode == "structural":
        for key, _ in CONE_COLORS:                       # fantasmas (removidos)
            g = ghosts.get(key, np.zeros((0, 2)))
            if len(g):
                ax.scatter(g[:, 0], g[:, 1], s=26, marker="x",
                           color="0.72", linewidths=0.9, zorder=2)
        for key, col in CONE_COLORS:                      # restantes (sólidos)
            p = td.get(key, np.zeros((0, 2)))
            if len(p):
                ax.scatter(p[:, 0], p[:, 1], s=13, color=col, zorder=3)
        title = f"{track} — {frac*100:.0f}% dos cones removidos da pista"
    else:
        for key, _ in CONE_COLORS:                        # todos reais a cinza
            p = td.get(key, np.zeros((0, 2)))
            if len(p):
                ax.scatter(p[:, 0], p[:, 1], s=9, color="0.8", zorder=2)
        seen_sc = {key: ax.scatter([], [], s=42, color=c, edgecolors="black",
                                   linewidths=0.4, zorder=4)
                   for key, c in CONE_COLORS}
        title = f"{track} — perceção com {frac*100:.0f}% de cones perdidos"

    ax.set_aspect("equal")
    pad = 6
    ax.set_xlim(cl[:, 0].min() - pad, cl[:, 0].max() + pad)
    ax.set_ylim(cl[:, 1].min() - pad, cl[:, 1].max() + pad)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    (traj,) = ax.plot([], [], "-", color="crimson", lw=1.6, zorder=5)
    (car,) = ax.plot([], [], "o", color="black", ms=8, zorder=7)
    (head,) = ax.plot([], [], "-", color="black", lw=2.2, zorder=6)
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=10, family="monospace",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    return seen_sc, traj, car, head, hud


def make_gif(out_path, model_path, track, frac, seed, fps, stride, device,
             mode='perception'):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    xs, ys, ths, spd, seen, td, info, ghosts = rollout(
        model_path, track, frac, seed, mode, device=device)
    N = len(xs)
    reason = info.get("termination_reason", "") or "fim"
    failed = info.get("laps_completed", 0) < 1

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    seen_sc, traj, car, head, hud = _draw_scene(ax, td, track, frac, mode, ghosts)

    def update(i):
        traj.set_data(xs[:i + 1], ys[:i + 1])
        car.set_data([xs[i]], [ys[i]])
        L = 3.5
        head.set_data([xs[i], xs[i] + L * np.cos(ths[i])],
                      [ys[i], ys[i] + L * np.sin(ths[i])])
        for key, _ in CONE_COLORS:                    # só atualiza no modo perception
            if key in seen_sc:
                pts = seen[i][key]
                seen_sc[key].set_offsets(pts if len(pts) else np.empty((0, 2)))
        n_seen = sum(len(seen[i][key]) for key, _ in CONE_COLORS)
        last = i >= N - 1 - stride
        status = (f"FALHA: {reason}" if (last and failed) else
                  ("VOLTA COMPLETA" if last else "a conduzir…"))
        hud.set_text(f"step {i:>4}\n{spd[i]:>4.0f} km/h\ncones vistos: {n_seen}\n{status}")
        hud.set_color("crimson" if (last and failed) else "black")
        return (traj, car, head, hud, *seen_sc.values())

    frames = list(range(0, N, stride)) + [N - 1] * max(1, fps // 2)
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=80)
    update(N - 1)
    fig.savefig(out_path.replace(".gif", "_final.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"[OK] {out_path}  (modo={mode}, laps={info.get('laps_completed')}, "
          f"cones={info.get('cones_hit')}, motivo='{reason}', steps={N})")
    return out_path


def make_contrast(out_path, model_path, track, frac_hi, seed, device,
                  mode='perception'):
    """Painel 2x: baseline vs degradação alta. baseline = 0% (structural) / 5% (perception)."""
    import matplotlib.pyplot as plt

    base = 0.0 if mode == "structural" else 0.05
    base_tag = ("0% (pista completa)" if mode == "structural"
                else "5% (sensor normal)")
    hi_tag = (f"{frac_hi*100:.0f}% (cones removidos)" if mode == "structural"
              else f"{frac_hi*100:.0f}% (falha de perceção)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, fr, tag in [(axes[0], base, base_tag), (axes[1], frac_hi, hi_tag)]:
        xs, ys, ths, spd, seen, td, info, ghosts = rollout(
            model_path, track, fr, seed, mode, device=device)
        N = len(xs)
        seen_sc, traj, car, head, hud = _draw_scene(ax, td, track, fr, mode, ghosts)
        mid = N // 2
        traj.set_data(xs, ys)
        car.set_data([xs[mid]], [ys[mid]])
        head.set_data([xs[mid], xs[mid] + 3.5 * np.cos(ths[mid])],
                      [ys[mid], ys[mid] + 3.5 * np.sin(ths[mid])])
        for key, _ in CONE_COLORS:
            if key in seen_sc:
                pts = seen[mid][key]
                seen_sc[key].set_offsets(pts if len(pts) else np.empty((0, 2)))
        ok = info.get("laps_completed", 0) >= 1
        reason = info.get("termination_reason", "") or "fim"
        ax.set_title(f"{track} — {tag}")
        hud.set_text(f"{'VOLTA OK' if ok else 'FALHA: ' + reason}\n"
                     f"cones vistos @meio: "
                     f"{sum(len(seen[mid][k]) for k, _ in CONE_COLORS)}")
        hud.set_color("black" if ok else "crimson")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[OK] {out_path}.png (+ .pdf)  (modo={mode})")


def main():
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="perception",
                    choices=["perception", "structural"])
    ap.add_argument("--model", default="runs/sac_seed2/best/best_model.zip")
    ap.add_argument("--track", default="FSI24")
    ap.add_argument("--dropout", type=float, default=0.6,
                    help="perception: perda/frame; structural: fração removida.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--contrast", action="store_true")
    args = ap.parse_args()

    default_name = ("perception_failure" if args.mode == "perception"
                    else "structural_failure")
    out = args.out or os.path.join("results", "figures", f"{default_name}.gif")

    make_gif(out, args.model, args.track, args.dropout, args.seed,
             args.fps, args.stride, args.device, mode=args.mode)
    if args.contrast:
        make_contrast(os.path.join("results", "figures", f"{default_name}_contrast"),
                      args.model, args.track, args.dropout, args.seed,
                      args.device, mode=args.mode)


if __name__ == "__main__":
    main()
