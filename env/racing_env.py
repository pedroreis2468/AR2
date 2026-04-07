"""
Ambiente Gymnasium para corrida Formula Student com cones.

Merge de ambas as versões:
  - Persistent cone knockdowns (AR) — cones derrubados desaparecem do FOV
  - DOO/OC rules FS-AI realistas (AR) — timer off-course, limites configuráveis
  - Reward refinado (MEU) — progresso x2, alignment, dead-band smoothness,
    lateral linear/quadrática, stagnation checkpoint
  - Track loader YAML (MEU) — pistas reais de competição FS
  - Orange cones na observação (MEU) — obs_dim 27
  - 6-dim ego state (MEU) — inclui ay
  - effective_hw dinâmico (MEU) — track_width variável por pista

Observação (27 dims com orange cones, 21 sem):
    [0]     vx         - velocidade longitudinal normalizada
    [1]     vy         - velocidade lateral normalizada
    [2]     omega      - yaw rate normalizado
    [3]     steering   - ângulo de viragem atual normalizado
    [4]     ax         - aceleração longitudinal normalizada
    [5]     ay         - aceleração lateral normalizada
    [6:12]  blue_cones - 3 cones azuis mais próximos (x,y) no ref. do carro
    [12:18] yellow_cones - 3 cones amarelos mais próximos (x,y)
    [18:24] orange_cones - 3 cones laranja (se use_orange_cones)
    [-3]    dist_left  - distância à fronteira esquerda normalizada
    [-2]    dist_right - distância à fronteira direita normalizada
    [-1]    heading_err - erro de heading normalizado

Ação (2 dims contínuas):
    [0] steering_cmd  in [-1, 1]
    [1] throttle_cmd  in [-1, 1] (negativo = travão)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple

from .car_model import KinematicBicycleModel, VehicleParams
from .track_generator import TrackGenerator, TrackParams
from .track_loader import YAMLTrackLoader
from .cone_sensor import ConeSensor, BoundarySensor


class FSRacingEnv(gym.Env):

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        track_seed: Optional[int] = None,
        randomize_track: bool = True,
        max_episode_steps: int = 5000,
        vehicle_params: Optional[VehicleParams] = None,
        track_params: Optional[TrackParams] = None,
        domain_randomization: bool = True,
        terminate_on_cone: bool = False,
        dt: float = 0.02,
        action_repeat: int = 2,
        tracks_dir: str = None,
        track_name: Optional[str] = None,
        use_orange_cones: bool = True,
        cone_penalty_reward: float = -8.0,
        cone_penalty_seconds: float = 2.0,
        orange_cone_penalty_reward: float = -16.0,
        doo_cone_limit: int = 10,
        oc_lateral_limit: float = None,
        oc_extreme_limit: float = 5.0,
        oc_time_limit: float = 2.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.randomize_track = randomize_track
        self.max_episode_steps = max_episode_steps
        self.domain_randomization = domain_randomization
        self.terminate_on_cone = terminate_on_cone
        self.dt = dt
        self.action_repeat = action_repeat

        self.cone_penalty_reward = cone_penalty_reward
        self.cone_penalty_seconds = cone_penalty_seconds
        self.orange_cone_penalty_reward = orange_cone_penalty_reward
        self.doo_cone_limit = doo_cone_limit
        self.oc_extreme_limit = oc_extreme_limit
        self.oc_time_limit = oc_time_limit
        self._oc_lateral_limit_override = oc_lateral_limit

        self.vehicle_params = vehicle_params or VehicleParams()
        self.track_params = track_params or TrackParams()
        self.track_seed = track_seed
        self.tracks_dir = tracks_dir
        self.track_name = track_name
        self.current_track_name = track_name or ''

        self.car = KinematicBicycleModel(self.vehicle_params, dt)
        self.cone_sensor = ConeSensor()
        self.boundary_sensor = BoundarySensor()

        if tracks_dir is not None:
            self.track_loader = YAMLTrackLoader(tracks_dir)
            self._yaml_rng = np.random.RandomState(track_seed)
            self.track_gen = None
            print(f"[INFO] Carregadas {self.track_loader.n_tracks} pistas YAML: "
                  f"{', '.join(self.track_loader.track_names)}")
        else:
            self.track_loader = None
            self.track_gen = TrackGenerator(self.track_params, track_seed)
            print("[INFO] A usar track generator procedural")

        self.use_orange_cones = use_orange_cones
        self.obs_dim = 27 if use_orange_cones else 21
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32)

        self.track_data = None
        self.current_step = 0
        self.total_progress = 0.0
        self.prev_progress = 0.0
        self.prev_cl_idx = 0
        self.laps_completed = 0
        self.episode_reward = 0.0
        self.prev_action = np.zeros(2)

        self.knocked_blue = set()
        self.knocked_yellow = set()
        self.knocked_orange = set()
        self.total_cones_hit = 0
        self.total_time_penalty = 0.0
        self.oc_timer = 0.0

        self._progress_checkpoint = 0.0
        self._progress_checkpoint_step = 0
        self._renderer = None

    # ─── Reset ───────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_track or self.track_data is None:
            if self.track_loader is not None:
                if self.track_name:
                    self.track_data = self.track_loader.load_track(name=self.track_name)
                    self.current_track_name = self.track_name
                else:
                    self.track_data, self.current_track_name = \
                        self.track_loader.load_random(self._yaml_rng)
            else:
                if seed is not None:
                    self.track_gen.rng = np.random.RandomState(seed)
                self.track_data = self.track_gen.generate()
                self.current_track_name = 'procedural'

        td = self.track_data
        start_offset = 0.0
        v_init = 0.0
        if self.domain_randomization:
            start_offset = np.random.uniform(-0.5, 0.5) * self.track_params.track_width * 0.3
            v_init = np.random.uniform(0.0, 3.0)
            self._randomize_vehicle()

        start_pos = td['start_pos'].copy()
        start_pos += td['normals'][0] * start_offset
        self.car.reset(x=start_pos[0], y=start_pos[1], theta=td['start_heading'], v=v_init)

        self.current_step = 0
        self.total_progress = 0.0
        self.prev_progress = 0.0
        self.prev_cl_idx = 0
        self.laps_completed = 0
        self.episode_reward = 0.0
        self.prev_action = np.zeros(2)
        self._trajectory = [(self.car.x, self.car.y)]
        self.knocked_blue = set()
        self.knocked_yellow = set()
        self.knocked_orange = set()
        self.total_cones_hit = 0
        self.total_time_penalty = 0.0
        self.oc_timer = 0.0
        self._progress_checkpoint = 0.0
        self._progress_checkpoint_step = 0

        effective_hw = self._get_effective_hw()
        self.oc_lateral_limit = self._oc_lateral_limit_override if self._oc_lateral_limit_override else effective_hw + 0.5

        return self._get_obs(), self._get_info()

    # ─── Step ────────────────────────────────────────────────────────────

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        for _ in range(self.action_repeat):
            self.car.step(action)
        self.current_step += 1
        self._trajectory.append((self.car.x, self.car.y))
        reward, terminated = self._compute_reward(action)
        truncated = self.current_step >= self.max_episode_steps
        self.prev_action = action.copy()
        self.episode_reward += reward
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ─── Observação (MEU 6-dim ego + orange cones + AR active cones) ─────

    def _get_obs(self):
        td = self.track_data
        p = self.vehicle_params
        state = self.car.get_state()
        max_lat_accel = p.mu * 9.81

        ego = np.array([
            state['v'] / p.max_speed,
            (state['omega'] * p.lr) / (p.max_speed + 1e-6),
            state['omega'] / (p.max_speed / p.wheelbase),
            state['steering'] / p.max_steering,
            state['ax'] / p.max_accel,
            state['ay'] / max_lat_accel,
        ], dtype=np.float32)

        active_blue = self._get_active_cones('blue')
        active_yellow = self._get_active_cones('yellow')
        active_orange = self._get_active_cones('orange')

        if self.use_orange_cones:
            blue_obs, yellow_obs, orange_obs, _ = self.cone_sensor.get_observations(
                self.car.x, self.car.y, self.car.theta,
                active_blue, active_yellow, active_orange,
                add_noise=self.domain_randomization)
            cone_obs = np.concatenate([
                blue_obs.flatten() / self.cone_sensor.max_range,
                yellow_obs.flatten() / self.cone_sensor.max_range,
                orange_obs.flatten() / self.cone_sensor.max_range,
            ]).astype(np.float32)
        else:
            blue_obs, yellow_obs, _, _ = self.cone_sensor.get_observations(
                self.car.x, self.car.y, self.car.theta,
                active_blue, active_yellow, np.zeros((0, 2)),
                add_noise=self.domain_randomization)
            cone_obs = np.concatenate([
                blue_obs.flatten() / self.cone_sensor.max_range,
                yellow_obs.flatten() / self.cone_sensor.max_range,
            ]).astype(np.float32)

        effective_hw = self._get_effective_hw()
        boundary = self.boundary_sensor.get_boundary_info(
            self.car.x, self.car.y, self.car.theta,
            td['centerline'], td['tangents'], td['normals'], effective_hw)

        return np.clip(np.concatenate([ego, cone_obs, boundary]), -2.0, 2.0).astype(np.float32)

    # ─── Active cones — persistent knockdowns (AR) ───────────────────────

    def _get_active_cones(self, color):
        td = self.track_data
        if color == 'blue':
            all_c, knocked = td['blue_cones'], self.knocked_blue
        elif color == 'yellow':
            all_c, knocked = td['yellow_cones'], self.knocked_yellow
        else:
            all_c, knocked = td['orange_cones'], self.knocked_orange
        if not knocked:
            return all_c
        mask = np.ones(len(all_c), dtype=bool)
        for idx in knocked:
            if idx < len(all_c):
                mask[idx] = False
        active = all_c[mask]
        return active if len(active) > 0 else np.empty((0, 2), dtype=all_c.dtype)

    # ─── Reward (MEU shaping + AR FS-AI penalties) ───────────────────────

    def _compute_reward(self, action):
        td = self.track_data
        cl = td['centerline']
        n_cl = len(cl)

        car_pos = np.array([self.car.x, self.car.y])
        dists_sq = np.sum((cl - car_pos)**2, axis=1)
        cl_idx = int(np.argmin(dists_sq))
        lateral_dist = float(np.sqrt(dists_sq[cl_idx]))
        effective_hw = self._get_effective_hw()

        # Progresso com sinal correto (MEU)
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

        if (self.total_progress > td['track_length'] * 0.9
                and cl_idx < n_cl * 0.1 and self.prev_cl_idx > n_cl * 0.8):
            self.laps_completed += 1
            self.total_progress = 0.0
        self.prev_cl_idx = cl_idx

        heading_err = self._get_heading_error(cl_idx)

        # Reward components (MEU shaping)
        r_progress = progress * 2.0
        v_norm = self.car.v / (self.vehicle_params.max_speed + 1e-6)
        r_alignment = v_norm * float(np.cos(heading_err)) * 0.4
        steer_change = abs(float(action[0]) - float(self.prev_action[0]))
        r_smooth = -max(0.0, steer_change - 0.1) * 0.05
        lat_ratio = lateral_dist / (effective_hw + 1e-6)
        r_lateral = -lat_ratio * 0.08 if lat_ratio <= 1.0 else -0.08 - (lat_ratio - 1.0)**2 * 2.0
        r_time = -0.005

        reward = r_progress + r_alignment + r_smooth + r_lateral + r_time

        # Cone collisions — persistent (AR)
        reward += self._process_cone_collisions()
        if self.terminate_on_cone and self.total_cones_hit > 0:
            return float(reward), True

        # Off-course handling (AR)
        terminated = False
        step_time = self.dt * self.action_repeat
        if lateral_dist > self.oc_lateral_limit:
            self.oc_timer += step_time
            reward -= 0.5 * (lateral_dist / self.oc_lateral_limit) ** 2
        else:
            self.oc_timer = 0.0

        # DOO conditions (AR)
        if self.total_cones_hit >= self.doo_cone_limit:
            reward -= 50.0; terminated = True
        if lateral_dist > self.oc_extreme_limit:
            reward -= 50.0; terminated = True
        if self.oc_timer >= self.oc_time_limit:
            reward -= 30.0; terminated = True

        # Lap bonus (AR)
        if self.laps_completed > 0:
            reward += max(200.0 - self.total_cones_hit * 5.0, 50.0)
            terminated = True

        # Contramão
        if diff_idx < -5:
            reward -= 20.0; terminated = True

        # Estagnação (MEU)
        if self.car.v < 0.1 and self.current_step > 200:
            reward -= 0.1
        if self.current_step - self._progress_checkpoint_step >= 300:
            if self.total_progress - self._progress_checkpoint < 2.0:
                reward -= 5.0; terminated = True
            self._progress_checkpoint = self.total_progress
            self._progress_checkpoint_step = self.current_step

        self.prev_progress = self.total_progress
        return float(reward), terminated

    # ─── Persistent cone collisions (AR) ─────────────────────────────────

    def _process_cone_collisions(self):
        td = self.track_data
        
        # Pré-calcular trigonometria para o referencial local
        cos_t = np.cos(-self.car.theta)
        sin_t = np.sin(-self.car.theta)
        
        # Limites da Bounding Box (Metade do tamanho + margem de erro)
        half_length = (self.vehicle_params.length / 2.0) + 0.15
        half_width = (self.vehicle_params.width / 2.0) + 0.15

        penalty = 0.0
        for cones, knocked, pen, ts_mult in [
            (td['blue_cones'], self.knocked_blue, self.cone_penalty_reward, 1),
            (td['yellow_cones'], self.knocked_yellow, self.cone_penalty_reward, 1),
            (td['orange_cones'], self.knocked_orange, self.orange_cone_penalty_reward, 2),
        ]:
            for i, cone in enumerate(cones):
                if i not in knocked:
                    # Distância global
                    dx = cone[0] - self.car.x
                    dy = cone[1] - self.car.y
                    
                    # Rotação para o referencial do carro
                    local_x = dx * cos_t - dy * sin_t
                    local_y = dx * sin_t + dy * cos_t
                    
                    # Verificar colisão com a bounding box retangular
                    if abs(local_x) < half_length and abs(local_y) < half_width:
                        knocked.add(i)
                        self.total_cones_hit += 1
                        self.total_time_penalty += self.cone_penalty_seconds * ts_mult
                        penalty += pen
        return penalty

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _get_effective_hw(self):
        return self.track_data.get('track_width', self.track_params.track_width) / 2.0

    def _get_heading_error(self, cl_idx):
        t = self.track_data['tangents'][cl_idx]
        err = self.car.theta - np.arctan2(t[1], t[0])
        return (err + np.pi) % (2 * np.pi) - np.pi

    def _randomize_vehicle(self):
        p = self.vehicle_params
        p.mass = 250.0 * np.random.uniform(0.85, 1.15)
        p.mu = 1.5 * np.random.uniform(0.80, 1.20)
        p.drag_coeff = 0.02 * np.random.uniform(0.90, 1.10)

    def _get_info(self):
        return {
            'speed': self.car.v, 'speed_kmh': self.car.v * 3.6,
            'total_progress': self.total_progress, 'laps_completed': self.laps_completed,
            'episode_reward': self.episode_reward, 'step': self.current_step,
            'x': self.car.x, 'y': self.car.y, 'theta': self.car.theta,
            'steering': self.car.steering, 'track_name': self.current_track_name,
            'cones_hit': self.total_cones_hit,
            'cones_hit_blue': len(self.knocked_blue),
            'cones_hit_yellow': len(self.knocked_yellow),
            'cones_hit_orange': len(self.knocked_orange),
            'time_penalty': self.total_time_penalty,
            'off_course_timer': self.oc_timer, 'doo_limit': self.doo_cone_limit,
        }

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .renderer import FSRenderer
            self._renderer = FSRenderer(self)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register(id='FSRacing-v0', entry_point='env.racing_env:FSRacingEnv', max_episode_steps=5000)
