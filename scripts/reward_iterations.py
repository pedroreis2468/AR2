"""
reward_iterations.py — Estudo do design da recompensa (10 iteracoes).

Reconstrucao da PROGRESSAO de reward shaping do projeto: 10 estagios, cada um
mudando UMA coisa, com as magnitudes a variar de forma suave entre os extremos
REAIS documentados no git (v1 = commit 8c4cf3a; final = recompensa de producao).
Treina um agente SAC por estagio, avalia, e gera metricas + figuras que mostram
o carro a passar de "mal se mexe" para "completa voltas".

Honestidade: os pesos da v1 (estagio 1-ish) e do final (estagio 10) sao exatos
(do git). Os estagios intermedios sao uma reconstrucao plausivel da afinacao —
e um ESTUDO DE DESIGN, nao o registo verbatim dos commits. So a recompensa varia
entre estagios; tudo o resto (pista, sensor, colisao, regras) e mantido fixo
para isolar o efeito do shaping.

CONCEBIDO PARA CORRER NUMA GPU DECENTE (ex.: RTX 5070 Ti).

Uso tipico (treina os 10 + avalia + figuras):
    python scripts/reward_iterations.py --steps 150000 --n-envs 4 --device cuda

Outras opcoes:
    --only 1,2,10        treina apenas estes estagios (1-indexed)
    --skip-trained       salta estagios cujo modelo final ja existe (retomar)
    --eval-only          nao treina; so avalia/replota o que ja existe
    --eval-episodes 5    episodios por pista na avaliacao
    --eval-track FSG19   pista usada para a figura de trajetorias

Saidas:
    runs/reward_iter/stageNN_<slug>/final_model.zip (+ vecnormalize.pkl)
    results/reward_iter/metrics.csv
    results/figures/reward_iterations.{pdf,png}
    results/figures/reward_iter_trajectories.{pdf,png}
"""
import os
import sys
import csv
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.racing_env import FSRacingEnv
from env.track_splits import TRAIN_TRACKS

RUNS_DIR = os.path.join(ROOT, "runs", "reward_iter")
RES_DIR = os.path.join(ROOT, "results", "reward_iter")
FIG_DIR = os.path.join(ROOT, "results", "figures")

# ---------------------------------------------------------------------------
# Definicao dos 10 estagios. Cada cfg e a recompensa COMPLETA desse estagio.
# Magnitudes "suavizadas" entre v1 (do git) e final.
# ---------------------------------------------------------------------------
_BASE = dict(progress_w=1.0, align_w=0.0, smooth_w=0.0, smooth_db=0.1,
             lateral_mode="none", lateral_w=0.0, time_w=0.0, stagnation="none")


def _cfg(**upd):
    d = dict(_BASE); d.update(upd); return d


STAGES = [
    dict(slug="progress_only", name="Só progresso (×1.0)",
         sintoma="Sem incentivo a despachar nem a ficar na pista; mal progride / erra cedo.",
         cfg=_cfg(progress_w=1.0)),
    dict(slug="time_cost", name="+ custo de tempo (−0.01)",
         sintoma="Para não ficar parado, mas ainda sem noção de pista.",
         cfg=_cfg(progress_w=1.0, time_w=0.01)),
    dict(slug="align_weak", name="+ alinhamento fraco (×0.1)",
         sintoma="Começa a andar na direção certa, mas devagar e a cortar.",
         cfg=_cfg(progress_w=1.0, time_w=0.01, align_w=0.1)),
    dict(slug="progress_up", name="Reforçar progresso (×1.5)",
         sintoma="Avança mais decidido, mas sai muito da pista.",
         cfg=_cfg(progress_w=1.5, time_w=0.01, align_w=0.1)),
    dict(slug="lateral_quad", name="+ lateral quadrática (×0.2)",
         sintoma="Tenta ficar na pista, mas penalização agressiva ⇒ conservador.",
         cfg=_cfg(progress_w=1.5, time_w=0.01, align_w=0.1, lateral_mode="quad", lateral_w=0.2)),
    dict(slug="lateral_piece", name="Lateral suavizada (linear dentro / quad. fora)",
         sintoma="Passa a usar a largura da pista sem ser sufocado no centro.",
         cfg=_cfg(progress_w=1.5, time_w=0.01, align_w=0.1, lateral_mode="piece", lateral_w=0.08)),
    dict(slug="smooth_nodb", name="+ suavidade SEM dead-band (−0.3·|Δδ|)",
         sintoma="REGRESSÃO: penalizar toda a viragem bloqueia as curvas.",
         cfg=_cfg(progress_w=1.5, time_w=0.01, align_w=0.1, lateral_mode="piece", lateral_w=0.08,
                  smooth_w=0.3, smooth_db=0.0)),
    dict(slug="smooth_deadband", name="Suavidade com dead-band 0.1 (×0.05)",
         sintoma="Correção: viragem normal fica livre, só penaliza solavancos.",
         cfg=_cfg(progress_w=1.5, time_w=0.01, align_w=0.1, lateral_mode="piece", lateral_w=0.08,
                  smooth_w=0.05, smooth_db=0.1)),
    dict(slug="rescale_final", name="Reescala dos sinais (progr.×2.0, align×0.4, tempo−0.005)",
         sintoma="Progresso torna-se o sinal dominante; mais rápido e estável.",
         cfg=_cfg(progress_w=2.0, time_w=0.005, align_w=0.4, lateral_mode="piece", lateral_w=0.08,
                  smooth_w=0.05, smooth_db=0.1)),
    dict(slug="anti_stagnation", name="+ anti-estagnação (checkpoint DOO) = FINAL",
         sintoma="Acaba com episódios 'zombie' parados; recompensa de produção.",
         cfg=_cfg(progress_w=2.0, time_w=0.005, align_w=0.4, lateral_mode="piece", lateral_w=0.08,
                  smooth_w=0.05, smooth_db=0.1, stagnation="full")),
]


# ---------------------------------------------------------------------------
# Ambiente com recompensa configuravel (so muda o shaping; resto = FSRacingEnv)
# ---------------------------------------------------------------------------
class StagedRewardEnv(FSRacingEnv):
    """FSRacingEnv com _compute_reward parametrizado por um reward_cfg.

    Mantem identica toda a logica nao-shaping (progresso, deteccao de volta,
    off-course, DOO, bonus de volta, contramao). So os termos de shaping e a
    anti-estagnacao sao controlados pela cfg.
    """

    def __init__(self, *args, reward_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rcfg = dict(_BASE)
        if reward_cfg:
            self.rcfg.update(reward_cfg)

    def _compute_reward(self, action):
        c = self.rcfg
        td = self.track_data
        cl = td['centerline']
        n_cl = len(cl)

        car_pos = np.array([self.car.x, self.car.y])
        dists_sq = np.sum((cl - car_pos) ** 2, axis=1)
        cl_idx = int(np.argmin(dists_sq))
        lateral_dist = float(np.sqrt(dists_sq[cl_idx]))
        effective_hw = self._get_effective_hw()

        # Progresso com sinal (identico ao env base)
        diff_idx = (cl_idx - self.prev_cl_idx + n_cl) % n_cl
        if diff_idx > n_cl // 2:
            diff_idx -= n_cl
        progress = 0.0
        if diff_idx > 0:
            for i in range(diff_idx):
                a = (self.prev_cl_idx + i) % n_cl
                b = (self.prev_cl_idx + i + 1) % n_cl
                progress += float(np.linalg.norm(cl[b] - cl[a]))
        elif diff_idx < 0:
            for i in range(-diff_idx):
                a = (self.prev_cl_idx - i) % n_cl
                b = (self.prev_cl_idx - i - 1) % n_cl
                progress -= float(np.linalg.norm(cl[b] - cl[a]))
        self.total_progress += max(0.0, progress)

        # Deteccao de volta (identica ao env base)
        finish_idx = self._finish_cl_idx

        def _crossed_finish(prev_i, curr_i, finish_i, n):
            if prev_i == curr_i:
                return False
            d_prev = (prev_i - finish_i) % n
            d_curr = (curr_i - finish_i) % n
            return d_prev > n * 0.5 and d_curr < n * 0.5

        if (self.total_progress > td['track_length'] * 0.85
                and _crossed_finish(self.prev_cl_idx, cl_idx, finish_idx, n_cl)):
            self.laps_completed += 1
            self.total_progress = 0.0
            self._progress_checkpoint = 0.0
            self._progress_checkpoint_step = self.current_step
        self.prev_cl_idx = cl_idx

        heading_err = self._get_heading_error(cl_idx)
        v_norm = self.car.v / (self.vehicle_params.max_speed + 1e-6)

        # --- TERMOS DE SHAPING (controlados pela cfg) ---
        r_progress = progress * c["progress_w"]
        r_align = v_norm * float(np.cos(heading_err)) * c["align_w"]
        steer_change = abs(float(action[0]) - float(self.prev_action[0]))
        r_smooth = -max(0.0, steer_change - c["smooth_db"]) * c["smooth_w"]
        lat_ratio = lateral_dist / (effective_hw + 1e-6)
        if c["lateral_mode"] == "none":
            r_lateral = 0.0
        elif c["lateral_mode"] == "quad":
            r_lateral = -(lat_ratio ** 2) * c["lateral_w"]
        else:  # "piece": linear dentro, quadratica fora
            r_lateral = (-lat_ratio * c["lateral_w"] if lat_ratio <= 1.0
                         else -c["lateral_w"] - (lat_ratio - 1.0) ** 2 * 2.0)
        r_time = -c["time_w"]

        reward = r_progress + r_align + r_smooth + r_lateral + r_time

        # Colisoes com cones (identico)
        reward += self._process_cone_collisions()
        if self.terminate_on_cone and self.total_cones_hit > 0:
            return float(reward), True

        # Off-course (identico)
        terminated = False
        step_time = self.dt * self.action_repeat
        if lateral_dist > self.oc_lateral_limit:
            self.oc_timer += step_time
            reward -= 0.5 * (lateral_dist / self.oc_lateral_limit) ** 2
        else:
            self.oc_timer = 0.0

        # DOO (identico)
        if self.total_cones_hit >= self.doo_cone_limit:
            reward -= 50.0; terminated = True
        if lateral_dist > self.oc_extreme_limit:
            reward -= 50.0; terminated = True
            self._termination_reason = f'off-course extremo ({lateral_dist:.1f}m)'
        if self.oc_timer >= self.oc_time_limit:
            reward -= 30.0; terminated = True
            self._termination_reason = f'off-course timeout ({self.oc_timer:.1f}s)'

        # Bonus de volta (identico)
        if self.laps_completed > self._last_rewarded_lap:
            reward += max(200.0 - self.total_cones_hit * 5.0, 50.0)
            self._last_rewarded_lap = self.laps_completed
            self._termination_reason = f'volta completa ({self.laps_completed}/{self.max_laps})'
            if self.laps_completed >= self.max_laps:
                terminated = True

        # Contramao (identico)
        if diff_idx < -5:
            reward -= 20.0; terminated = True
            self._termination_reason = f'contramao (diff_idx={diff_idx})'

        # Anti-estagnacao (controlada pela cfg)
        if c["stagnation"] == "full":
            if self.car.v < 0.1 and self.current_step > 200:
                reward -= 0.1
            if self.current_step - self._progress_checkpoint_step >= 300:
                if self.total_progress - self._progress_checkpoint < 2.0:
                    reward -= 5.0; terminated = True
                    self._termination_reason = 'estagnacao (<2m em 300 steps)'
                self._progress_checkpoint = self.total_progress
                self._progress_checkpoint_step = self.current_step

        self.prev_progress = self.total_progress
        return float(reward), terminated


# ---------------------------------------------------------------------------
# Treino de um estagio
# ---------------------------------------------------------------------------
def train_stage(idx, stage, args):
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    run_dir = os.path.join(RUNS_DIR, f"stage{idx:02d}_{stage['slug']}")
    os.makedirs(run_dir, exist_ok=True)

    def make_env(rank):
        def _init():
            env = StagedRewardEnv(
                reward_cfg=stage["cfg"],
                randomize_track=True, domain_randomization=True,
                max_episode_steps=args.max_ep_steps, terminate_on_cone=True,
                track_seed=args.seed * 1000 + rank, tracks_dir=os.path.join(ROOT, "tracks"),
                allowed_tracks=TRAIN_TRACKS, use_orange_cones=True,
            )
            return Monitor(env, os.path.join(run_dir, f"monitor_{rank}"))
        return _init

    n = args.n_envs
    venv = (SubprocVecEnv([make_env(i) for i in range(n)]) if n > 1
            else DummyVecEnv([make_env(0)]))
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=50.0, gamma=0.99)

    model = SAC(
        "MlpPolicy", venv, device=args.device, seed=args.seed,
        learning_rate=3e-4, buffer_size=args.buffer_size, batch_size=256,
        gamma=0.99, tau=0.005, learning_starts=1000, target_entropy=-0.5,
        ent_coef="auto", train_freq=1, gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256], optimizer_kwargs=dict(eps=1e-5)),
        verbose=0,
    )
    print(f"\n[{idx}/10] Treino: {stage['slug']} — {stage['name']}")
    print(f"        cfg={stage['cfg']}")
    model.learn(total_timesteps=args.steps, progress_bar=True)
    model.save(os.path.join(run_dir, "final_model"))
    venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
    venv.close()
    print(f"        -> guardado em {os.path.relpath(run_dir, ROOT)}")
    return run_dir


# ---------------------------------------------------------------------------
# Avaliacao de um estagio (metricas + 1 trajetoria na pista fixa)
# ---------------------------------------------------------------------------
def eval_stage(idx, stage, args):
    from stable_baselines3 import SAC
    run_dir = os.path.join(RUNS_DIR, f"stage{idx:02d}_{stage['slug']}")
    model_path = os.path.join(run_dir, "final_model.zip")
    if not os.path.exists(model_path):
        print(f"[{idx}] sem modelo treinado em {model_path} — salto avaliacao.")
        return None, None
    model = SAC.load(model_path, device=args.device)

    def make(track):
        return StagedRewardEnv(
            reward_cfg=stage["cfg"], render_mode=None, randomize_track=False,
            domain_randomization=False, max_episode_steps=args.max_ep_steps,
            tracks_dir=os.path.join(ROOT, "tracks"), track_name=track,
            use_orange_cones=True, terminate_on_cone=False, doo_cone_limit=999, max_laps=1)

    speeds, progs, laps, cones, lens, statn = [], [], [], [], [], []
    for track in args.eval_tracks:
        for ep in range(args.eval_episodes):
            env = make(track)
            obs, info = env.reset(seed=ep)
            done = False; n_slow = 0; steps = 0
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc; steps += 1
                if env.car.v < 0.5:
                    n_slow += 1
            speeds.append(info["speed_kmh"]); progs.append(info["total_progress"])
            laps.append(info["laps_completed"]); cones.append(info["cones_hit"])
            lens.append(steps); statn.append(n_slow / max(1, steps))
            env.close()

    metrics = dict(
        stage=idx, slug=stage["slug"], name=stage["name"],
        lap_rate=100.0 * np.mean([l >= 1 for l in laps]),
        mean_speed_kmh=float(np.mean(speeds)), mean_progress_m=float(np.mean(progs)),
        frac_stationary=float(np.mean(statn)), mean_ep_len=float(np.mean(lens)),
        mean_cones=float(np.mean(cones)),
    )

    # 1 trajetoria deterministica na pista fixa (para a figura de trajetorias)
    env = make(args.eval_track)
    obs, info = env.reset(seed=0)
    xs, ys = [env.car.x], [env.car.y]; done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        xs.append(env.car.x); ys.append(env.car.y)
    traj = (np.array(xs), np.array(ys), env.track_data)
    env.close()
    print(f"[{idx}] {stage['slug']:18s} lap={metrics['lap_rate']:5.0f}% "
          f"vel={metrics['mean_speed_kmh']:5.1f}km/h prog={metrics['mean_progress_m']:6.1f}m "
          f"parado={100*metrics['frac_stationary']:4.0f}%")
    return metrics, traj


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def plot_progression(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["stage"])
    x = [r["stage"] for r in rows]
    labels = [r["slug"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    series = [("lap_rate", "Lap rate (%)", "#2ca02c"),
              ("mean_progress_m", "Progresso médio (m)", "#1f77b4"),
              ("frac_stationary", "Fração do tempo parado", "#d62728")]
    for ax, (key, ylab, col) in zip(axes, series):
        y = [r[key] for r in rows]
        if key == "frac_stationary":
            y = [v * 100 for v in y]; ylab = "Tempo parado (%)"
        ax.plot(x, y, "o-", color=col, lw=1.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    axes[1].set_title("Evolução ao longo das 10 iterações de reward shaping")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {out}.pdf")


def plot_trajectories(trajs, out, track_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = [(i, t) for i, t in trajs if t is not None]
    if not items:
        return
    td = items[0][1][2]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cl = td["centerline"]
    ax.plot(cl[:, 0], cl[:, 1], "--", color="0.8", lw=0.8)
    for key, c in [("blue_cones", "tab:blue"), ("yellow_cones", "gold"), ("orange_cones", "darkorange")]:
        p = td.get(key, np.zeros((0, 2)))
        if len(p):
            ax.scatter(p[:, 0], p[:, 1], s=8, color=c, zorder=1)
    cmap = plt.cm.viridis
    for i, (xs, ys, _) in items:
        ax.plot(xs, ys, "-", lw=1.6, color=cmap((i - 1) / 9.0), label=f"{i}", zorder=3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Trajetória por iteração em {track_name}\n(cor = iteração 1→10)")
    ax.legend(title="iter", fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {out}.pdf")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Estudo iterativo de reward shaping (10 estagios)")
    ap.add_argument("--steps", type=int, default=150000, help="steps de treino por estagio")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buffer-size", type=int, default=200000)
    ap.add_argument("--max-ep-steps", type=int, default=1500)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--eval-tracks", nargs="+", default=["FSG19", "FSG23", "FSS19"],
                    help="pistas (do train split) para metricas de avaliacao")
    ap.add_argument("--eval-track", type=str, default="FSG19",
                    help="pista para a figura de trajetorias")
    ap.add_argument("--only", type=str, default=None,
                    help="treina apenas estes estagios, ex.: 1,2,10 (1-indexed)")
    ap.add_argument("--skip-trained", action="store_true",
                    help="salta estagios cujo final_model.zip ja existe")
    ap.add_argument("--eval-only", action="store_true",
                    help="nao treina; so avalia/replota o que existe")
    args = ap.parse_args()

    import random
    random.seed(args.seed); np.random.seed(args.seed)
    try:
        import torch; torch.manual_seed(args.seed)
    except ImportError:
        pass

    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    only = set(int(x) for x in args.only.split(",")) if args.only else None

    # --- treino ---
    if not args.eval_only:
        for idx, stage in enumerate(STAGES, start=1):
            if only and idx not in only:
                continue
            run_dir = os.path.join(RUNS_DIR, f"stage{idx:02d}_{stage['slug']}")
            if args.skip_trained and os.path.exists(os.path.join(run_dir, "final_model.zip")):
                print(f"[{idx}] ja treinado — salto.")
                continue
            train_stage(idx, stage, args)

    # --- avaliacao + figuras (sempre sobre todos os estagios com modelo) ---
    print("\n=== Avaliacao ===")
    rows, trajs = [], []
    for idx, stage in enumerate(STAGES, start=1):
        m, t = eval_stage(idx, stage, args)
        if m is not None:
            rows.append(m); trajs.append((idx, t))

    if rows:
        csv_path = os.path.join(RES_DIR, "metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n[OK] {os.path.relpath(csv_path, ROOT)}")
        plot_progression(rows, os.path.join(FIG_DIR, "reward_iterations"))
        plot_trajectories(trajs, os.path.join(FIG_DIR, "reward_iter_trajectories"), args.eval_track)
    else:
        print("Nenhum estagio avaliado (treina primeiro).")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
