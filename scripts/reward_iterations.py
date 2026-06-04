"""
reward_iterations.py — Ablacao de reward: a jornada de fixes da recompensa (6 estagios REAIS).

Reconstroi a PROGRESSAO real de fixes do projeto, ancorada no historico git.
Cada estagio = um fix documentado, mudando ~UMA coisa, do ponto de partida
"suicide policy" (o carro anda em frente e morre) ate a recompensa de producao.
Treina um agente SAC por estagio, avalia, e gera metricas + figuras.

Ancoragem no git (verificado):
  - v0 = commit 9ac6a6b — "suicide policy": morte instantanea ao passar 1.3x da
    meia-largura (reward=-100), steering penalizado a -0.3 SEM dead-band,
    progresso so positivo. O carro morre em ~50 steps e nunca aprende a virar.
  - v1 = commit 8c4cf3a — primeiro fix: off-course passa a GRADUAL (sem morte
    instantanea); resto da recompensa ainda igual ao v0.
  - final = recompensa + treino de producao (estado atual do repo).

E uma ABLACAO DE REWARD: o treino e mantido FIXO (_TRAIN_OLD) em todos os
estagios para isolar o efeito da recompensa. Os fixes de TREINO (learning_starts,
target_entropy, norm_reward) sao uma questao separada — estao afinados para o
orcamento de producao (~3M steps) e a treino curto nem aquecem — por isso NAO
entram nesta comparacao. Pista/sensor/colisao/regras ficam fixos.

CONCEBIDO PARA CORRER NUMA GPU DECENTE (ex.: RTX 5070 Ti).

Uso tipico (treina os 6 estagios x N seeds + avalia + figuras):
    python scripts/reward_iterations.py --steps 500000 --n-seeds 3 --n-envs 4 --device cuda

Outras opcoes:
    --only 1,2,6         treina apenas estes estagios (1-indexed)
    --skip-trained       salta estagios cujo modelo final ja existe (retomar)
    --eval-only          nao treina; so avalia/replota o que ja existe
    --eval-episodes 5    episodios por pista na avaliacao
    --eval-track FSG19   pista usada para a figura de trajetorias

Saidas:
    runs/reward_iter/stageNN_<slug>/seedM/final_model.zip (+ vecnormalize.pkl, eval/curve.npz)
    results/reward_iter/metrics.csv
    results/figures/reward_iterations.{pdf,png}        (dashboard: velocidade / % a mexer / pista)
    results/figures/reward_iter_curves.{pdf,png}       (curvas de aprendizagem por iteracao)
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

def _std(v):
    """Desvio-padrao amostral (ddof=1); 0 se < 2 amostras."""
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def _sem(std, n):
    """Erro-padrao da media: std / sqrt(n). Incerteza DA MEDIA (nao do espalhamento
    dos seeds), apropriada para comparar iteracoes; ~2.5x menor que sigma com n=6."""
    import math
    std = np.asarray(std, dtype=float)
    return std / math.sqrt(n) if n >= 2 else np.zeros_like(std)

# ---------------------------------------------------------------------------
# Definicao dos 6 estagios (ablacao de reward, treino fixo). Cada cfg e a reward COMPLETA.
# Magnitudes "suavizadas" entre v1 (do git) e final.
# ---------------------------------------------------------------------------
_BASE = dict(progress_w=1.0, align_w=0.0, smooth_w=0.0, smooth_db=0.1,
             lateral_mode="none", lateral_w=0.0, time_w=0.0, stagnation="none",
             offtrack_mode="gradual")


def _cfg(**upd):
    d = dict(_BASE); d.update(upd); return d


# Configuracao de TREINO por estagio (alguns fixes nao sao da reward).
#   _TRAIN_OLD = v0/v1 (commit 9ac6a6b / 8c4cf3a): SAC "cru", entropia fixa,
#                episodios longos, sem normalizacao de reward.
#   _TRAIN_NEW = producao: arranque mais cedo, entropia guiada, episodios curtos,
#                VecNormalize(norm_reward).
_TRAIN_OLD = dict(learning_starts=5000, ent_coef=0.01, target_entropy="auto",
                  max_ep_steps=5000, norm_reward=False)
_TRAIN_NEW = dict(learning_starts=1000, ent_coef="auto", target_entropy=-0.5,
                  max_ep_steps=1500, norm_reward=True)


# Pesos de partida (v0, commit 9ac6a6b): align 0.1, smooth -0.3 SEM dead-band,
# lateral quadratica 0.2, time -0.01.
_V0 = dict(progress_w=1.0, align_w=0.1, smooth_w=0.3, smooth_db=0.0,
           lateral_mode="quad", lateral_w=0.2, time_w=0.01)


STAGES = [
    dict(slug="suicide_v0", name="v0 — suicide policy (morte instantânea 1.3×hw)",
         sintoma="Anda só em frente e morre ao sair 1.3× da meia-largura (~50 steps); nunca vive o suficiente para aprender a virar.",
         cfg=_cfg(offtrack_mode="instant13", **_V0), train=_TRAIN_OLD),
    dict(slug="offtrack_gradual", name="v1 — off-course gradual (acaba a morte instantânea)",
         sintoma="Já não morre ao tocar a borda; vive muito mais tempo, mas o steering −0.3 ainda bloqueia as curvas.",
         cfg=_cfg(offtrack_mode="gradual", **_V0), train=_TRAIN_OLD),
    dict(slug="steer_deadband", name="Steering −0.3 → −0.05 com dead-band 0.1",
         sintoma="A penalização deixa de sufocar a viragem normal: o carro COMEÇA A VIRAR.",
         cfg=_cfg(offtrack_mode="gradual", **{**_V0, "smooth_w": 0.05, "smooth_db": 0.1}),
         train=_TRAIN_OLD),
    dict(slug="lateral_piece", name="Lateral quad ×0.2 → piecewise ×0.08 (linear dentro/quad fora)",
         sintoma="Passa a usar a largura da pista sem ser sufocado no centro; deixa de ser conservador.",
         cfg=_cfg(offtrack_mode="gradual", progress_w=1.0, align_w=0.1, time_w=0.01,
                  smooth_w=0.05, smooth_db=0.1, lateral_mode="piece", lateral_w=0.08),
         train=_TRAIN_OLD),
    dict(slug="rescale_signals", name="Reescala dos sinais (progr.×2.0, align×0.4, tempo−0.005)",
         sintoma="Progresso torna-se dominante e o alinhamento puxa para rápido E alinhado.",
         cfg=_cfg(offtrack_mode="gradual", progress_w=2.0, align_w=0.4, time_w=0.005,
                  smooth_w=0.05, smooth_db=0.1, lateral_mode="piece", lateral_w=0.08),
         train=_TRAIN_OLD),
    dict(slug="anti_stagnation", name="+ anti-estagnação (checkpoint DOO 2m/300 steps) = reward FINAL",
         sintoma="Acaba com episódios 'zombie' parados; é a recompensa de produção.",
         cfg=_cfg(offtrack_mode="gradual", progress_w=2.0, align_w=0.4, time_w=0.005,
                  smooth_w=0.05, smooth_db=0.1, lateral_mode="piece", lateral_w=0.08,
                  stagnation="full"),
         train=_TRAIN_OLD),
]

# NOTA: esta e uma ABLACAO DE REWARD — o treino e mantido FIXO (_TRAIN_OLD) em
# todos os estagios para isolar o efeito da recompensa. Os "fixes de treino"
# (learning_starts, target_entropy, norm_reward) sao uma questao separada: estao
# afinados para o orcamento de producao (~3M steps) e a treino curto nem aquecem,
# pelo que nao entram nesta comparacao. O modelo de producao (treino completo)
# e avaliado a parte (ver notebook, seccoes 3-7).


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

        # Off-course
        terminated = False
        if c["offtrack_mode"] == "instant13":
            # BUG v0 (9ac6a6b): morte instantanea ao passar 1.3x da meia-largura.
            # O carro nunca vive o suficiente para aprender a virar.
            if lateral_dist > 1.3 * effective_hw:
                return -100.0, True
        else:
            # off-course gradual (v1+, identico ao env base)
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
# Callback: curva de aprendizagem por estagio (vs timesteps)
# ---------------------------------------------------------------------------
def _make_curve_callback(stage, args, run_dir):
    """Avalia periodicamente numa pista fixa e guarda a curva em
    run_dir/eval/curve.npz. Regista DUAS metricas comparaveis entre estagios:
      - lap_rate (%): fracao de episodios que fecham a volta;
      - completion (%): fracao da pista percorrida (1 volta = 100%). Usa
        laps_completed para nao sofrer com o reset de total_progress ao cruzar
        a meta — por isso e suave (0->100) e nao se inverte como o progresso cru.
    NAO usa a reward (cujos pesos variam entre estagios)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _Curve(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.eval_every = max(1500, args.steps // 20)
            self.n_eval = 3
            self._last = 0
            self.ts, self.lap, self.comp = [], [], []
            self._env = None

        def _eval_env(self):
            if self._env is None:
                self._env = StagedRewardEnv(
                    reward_cfg=stage["cfg"], render_mode=None,
                    randomize_track=False, domain_randomization=False,
                    max_episode_steps=1500, tracks_dir=os.path.join(ROOT, "tracks"),
                    track_name=args.eval_track, use_orange_cones=True,
                    terminate_on_cone=False, doo_cone_limit=999, max_laps=1)
            return self._env

        def _run_eval(self):
            env = self._eval_env()
            laps, comps = [], []
            for ep in range(self.n_eval):
                obs, info = env.reset(seed=ep)
                tl = float(env.track_data["track_length"])
                done = False
                while not done:
                    a, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = env.step(a)
                    done = term or trunc
                lp = info["laps_completed"]
                laps.append(lp)
                # 1 volta -> 100%; senao, fracao percorrida (sem reset)
                comps.append(1.0 if lp >= 1 else min(1.0, info["total_progress"] / tl))
            self.ts.append(int(self.num_timesteps))
            self.lap.append(100.0 * float(np.mean([l >= 1 for l in laps])))
            self.comp.append(100.0 * float(np.mean(comps)))

        def _on_step(self):
            if self.num_timesteps - self._last >= self.eval_every:
                self._last = self.num_timesteps
                self._run_eval()
            return True

        def _on_training_end(self):
            if not self.ts or self.ts[-1] != int(self.num_timesteps):
                self._run_eval()  # ponto final (evita duplicar o ultimo)
            eval_dir = os.path.join(run_dir, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            np.savez(os.path.join(eval_dir, "curve.npz"),
                     timesteps=np.array(self.ts), lap_rate=np.array(self.lap),
                     completion=np.array(self.comp))
            if self._env is not None:
                self._env.close()

    return _Curve()


# ---------------------------------------------------------------------------
# Treino de um estagio
# ---------------------------------------------------------------------------
def stage_dir(idx, stage):
    return os.path.join(RUNS_DIR, f"stage{idx:02d}_{stage['slug']}")


def seed_dir(idx, stage, seed):
    return os.path.join(stage_dir(idx, stage), f"seed{seed}")


def train_stage(idx, stage, args, seed):
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    run_dir = seed_dir(idx, stage, seed)
    os.makedirs(run_dir, exist_ok=True)

    # Config de treino do estagio (alguns fixes nao sao da reward).
    tcfg = dict(_TRAIN_NEW); tcfg.update(stage.get("train", {}))

    def make_env(rank):
        def _init():
            env = StagedRewardEnv(
                reward_cfg=stage["cfg"],
                randomize_track=True, domain_randomization=True,
                max_episode_steps=tcfg["max_ep_steps"], terminate_on_cone=True,
                track_seed=seed * 1000 + rank, tracks_dir=os.path.join(ROOT, "tracks"),
                allowed_tracks=TRAIN_TRACKS, use_orange_cones=True,
            )
            return Monitor(env, os.path.join(run_dir, f"monitor_{rank}"))
        return _init

    n = args.n_envs
    venv = (SubprocVecEnv([make_env(i) for i in range(n)]) if n > 1
            else DummyVecEnv([make_env(0)]))
    venv = VecNormalize(venv, norm_obs=False, norm_reward=tcfg["norm_reward"],
                        clip_reward=50.0, gamma=0.99)

    model = SAC(
        "MlpPolicy", venv, device=args.device, seed=seed,
        learning_rate=3e-4, buffer_size=args.buffer_size, batch_size=256,
        gamma=0.99, tau=0.005, learning_starts=tcfg["learning_starts"],
        target_entropy=tcfg["target_entropy"], ent_coef=tcfg["ent_coef"],
        train_freq=1, gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256], optimizer_kwargs=dict(eps=1e-5)),
        verbose=0,
    )
    print(f"\n[{idx}/{len(STAGES)}] seed {seed}: {stage['slug']} — {stage['name']}")
    print(f"        cfg={stage['cfg']}")
    print(f"        train={tcfg}")
    callback = _make_curve_callback(stage, args, run_dir)
    model.learn(total_timesteps=args.steps, progress_bar=True, callback=callback)
    model.save(os.path.join(run_dir, "final_model"))
    venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
    venv.close()
    print(f"        -> guardado em {os.path.relpath(run_dir, ROOT)}")
    return run_dir


# ---------------------------------------------------------------------------
# Avaliacao de um estagio (metricas + 1 trajetoria na pista fixa)
# ---------------------------------------------------------------------------
def _available_seeds(idx, stage, args):
    seeds = []
    for s in range(args.seed, args.seed + args.n_seeds):
        if os.path.exists(os.path.join(seed_dir(idx, stage, s), "final_model.zip")):
            seeds.append(s)
    return seeds


def eval_stage(idx, stage, args):
    from stable_baselines3 import SAC

    seeds = _available_seeds(idx, stage, args)
    if not seeds:
        print(f"[{idx}] sem modelos treinados em {stage_dir(idx, stage)} — salto avaliacao.")
        return None, None

    def make(track):
        return StagedRewardEnv(
            reward_cfg=stage["cfg"], render_mode=None, randomize_track=False,
            domain_randomization=False, max_episode_steps=args.max_ep_steps,
            tracks_dir=os.path.join(ROOT, "tracks"), track_name=track,
            use_orange_cones=True, terminate_on_cone=False, doo_cone_limit=999, max_laps=1)

    # metricas POR SEED (cada uma media sobre pistas x episodios), depois media+/-std
    per_seed = {k: [] for k in ("lap", "comp", "speed", "cones", "ep_len", "statn", "jerk")}
    for s in seeds:
        model = SAC.load(os.path.join(seed_dir(idx, stage, s), "final_model.zip"), device=args.device)
        laps, comps, speeds, cones, lens, statn, jerks = [], [], [], [], [], [], []
        for track in args.eval_tracks:
            for ep in range(args.eval_episodes):
                env = make(track)
                obs, info = env.reset(seed=ep)
                tl = float(env.track_data["track_length"])
                done = False; n_slow = 0; steps = 0
                prev_steer = None; dsum = 0.0
                while not done:
                    a, _ = model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = env.step(a)
                    done = term or trunc; steps += 1
                    if env.car.v < 0.5:
                        n_slow += 1
                    st = float(a[0])
                    if prev_steer is not None:
                        dsum += abs(st - prev_steer)
                    prev_steer = st
                lp = info["laps_completed"]
                laps.append(lp); comps.append(1.0 if lp >= 1 else min(1.0, info["total_progress"] / tl))
                speeds.append(info["speed_kmh"]); cones.append(info["cones_hit"])
                lens.append(steps); statn.append(n_slow / max(1, steps))
                jerks.append(dsum / max(1, steps - 1))   # |Δsteering| médio por step
                env.close()
        per_seed["lap"].append(100.0 * np.mean([l >= 1 for l in laps]))
        per_seed["comp"].append(100.0 * np.mean(comps))
        per_seed["speed"].append(float(np.mean(speeds)))
        per_seed["cones"].append(float(np.mean(cones)))
        per_seed["ep_len"].append(float(np.mean(lens)))
        per_seed["statn"].append(float(np.mean(statn)))
        per_seed["jerk"].append(float(np.mean(jerks)))

    metrics = dict(
        stage=idx, slug=stage["slug"], name=stage["name"], n_seeds=len(seeds),
        lap_rate=float(np.mean(per_seed["lap"])), lap_rate_std=_std(per_seed["lap"]),
        completion=float(np.mean(per_seed["comp"])), completion_std=_std(per_seed["comp"]),
        mean_speed_kmh=float(np.mean(per_seed["speed"])), mean_speed_kmh_std=_std(per_seed["speed"]),
        mean_cones=float(np.mean(per_seed["cones"])), mean_cones_std=_std(per_seed["cones"]),
        frac_stationary=float(np.mean(per_seed["statn"])), frac_stationary_std=_std(per_seed["statn"]),
        mean_jerk=float(np.mean(per_seed["jerk"])), mean_jerk_std=_std(per_seed["jerk"]),
        mean_ep_len=float(np.mean(per_seed["ep_len"])),
    )

    # 1 trajetoria deterministica (seed mais baixo) na pista fixa
    model = SAC.load(os.path.join(seed_dir(idx, stage, seeds[0]), "final_model.zip"), device=args.device)
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
    print(f"[{idx}] {stage['slug']:18s} ({len(seeds)} seeds) "
          f"lap={metrics['lap_rate']:5.0f}±{metrics['lap_rate_std']:<4.0f}% "
          f"compl={metrics['completion']:5.0f}% vel={metrics['mean_speed_kmh']:5.1f}km/h "
          f"cones={metrics['mean_cones']:.1f}")
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
    ns = rows[0].get("n_seeds", 1)
    # Lidera com as metricas LIMPAS/monotonas: velocidade e % a mexer.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    series = [("mean_speed_kmh", "mean_speed_kmh_std", "Velocidade média (km/h)", "#9467bd"),
              ("moving", "moving_std", "% do tempo a mexer", "#1f77b4"),
              ("completion", "completion_std", "Pista completada (%)", "#2ca02c")]
    for ax, (key, skey, ylab, col) in zip(axes, series):
        if key == "moving":
            y = [100.0 * (1.0 - r["frac_stationary"]) for r in rows]
            sd = [100.0 * r.get("frac_stationary_std", 0.0) for r in rows]
        else:
            y = [r[key] for r in rows]
            sd = [r.get(skey, 0.0) for r in rows]
        yerr = _sem(sd, ns)
        ax.errorbar(x, y, yerr=yerr, fmt="o-", color=col, lw=1.8, capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    axes[1].set_title(f"Ablação de reward — {len(rows)} iterações  (média ± erro-padrão, {ns} seeds)")
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
    n_iter = max(it for it, _ in items)
    for i, (xs, ys, _) in items:
        ax.plot(xs, ys, "-", lw=1.6, color=cmap((i - 1) / max(1, n_iter - 1)),
                label=f"{i}", zorder=3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Trajetória por iteração em {track_name}\n(cor = iteração 1→{n_iter})")
    ax.legend(title="iter", fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {out}.pdf")


def load_stage_curve(idx, stage, metric="completion"):
    """Le os curve.npz de todos os seeds de um estagio e devolve
    (timesteps, media, desvio) alinhados pelo menor comprimento comum.
    Devolve (None, None, None) se nao houver curvas."""
    import glob
    files = sorted(glob.glob(os.path.join(stage_dir(idx, stage), "seed*", "eval", "curve.npz")))
    series, ts_ref = [], None
    for f in files:
        d = np.load(f)
        if metric not in d:
            continue
        series.append((d["timesteps"], d[metric]))
    if not series:
        return None, None, None
    L = min(len(t) for t, _ in series)
    ts = series[0][0][:L]
    M = np.stack([v[:L] for _, v in series], axis=0)
    nseeds = M.shape[0]
    std = M.std(axis=0, ddof=1) if nseeds > 1 else np.zeros(L)
    return ts, M.mean(axis=0), _sem(std, nseeds)


def plot_learning_curves(out, metric="completion"):
    """Curvas (metric) vs timesteps, uma por estagio, media +/- IC 95% sobre seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cmap = plt.cm.viridis
    n = len(STAGES)
    any_curve = False
    for idx, stage in enumerate(STAGES, start=1):
        ts, mean, std = load_stage_curve(idx, stage, metric)
        if ts is None:
            continue
        any_curve = True
        c = cmap((idx - 1) / max(1, n - 1))
        ax.plot(ts, mean, "-o", ms=3, lw=1.8, color=c, label=f"{idx} {stage['slug']}")
        ax.fill_between(ts, mean - std, mean + std, color=c, alpha=0.13)
    if not any_curve:
        plt.close(fig)
        return
    ylab = {"completion": "Pista completada na avaliação (%)",
            "lap_rate": "Lap rate na avaliação (%)"}.get(metric, metric)
    ax.set_xlabel("Timesteps de treino"); ax.set_ylabel(ylab)
    ax.set_title("Curvas de aprendizagem por iteração (média ± erro-padrão sobre seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, title="iteração", loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {out}.pdf")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Ablacao de reward: jornada de fixes (6 estagios, treino fixo)")
    ap.add_argument("--steps", type=int, default=150000, help="steps de treino por estagio")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="seed base (usa seed..seed+n_seeds-1)")
    ap.add_argument("--n-seeds", type=int, default=1, help="nº de seeds por estagio (curvas média±σ)")
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

    seeds = list(range(args.seed, args.seed + args.n_seeds))

    # --- treino ---
    if not args.eval_only:
        for idx, stage in enumerate(STAGES, start=1):
            if only and idx not in only:
                continue
            for s in seeds:
                if args.skip_trained and os.path.exists(
                        os.path.join(seed_dir(idx, stage, s), "final_model.zip")):
                    print(f"[{idx}] seed {s} ja treinado — salto.")
                    continue
                train_stage(idx, stage, args, s)

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
    # curvas de aprendizagem (independentes da avaliacao; leem os curve.npz)
    plot_learning_curves(os.path.join(FIG_DIR, "reward_iter_curves"))
    print("\n[DONE]")


if __name__ == "__main__":
    main()
