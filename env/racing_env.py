"""
Ambiente Gymnasium para corrida Formula Student com cones.
Observação: estado do ego + cones + boundary info (~20 dims)
Ação: [steering, throttle] contínuos em [-1, 1]

Regras FS-AI realistas:
  - Cones derrubados aplicam penalização (2s cada), NÃO terminam o episódio
  - O carro pode passar além dos cones (off-track)
  - DOO (Did Not Operate) se demasiados cones derrubados (>10) ou off-course >5m
  - OC (Off-Course) se carro sai dos limites por 2+ segundos consecutivos
  - Cones laranja derrubados = penalidade agravada
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple

from .car_model import KinematicBicycleModel, VehicleParams
from .track_generator import TrackGenerator, TrackParams
from .cone_sensor import ConeSensor, BoundarySensor


class FSRacingEnv(gym.Env):
    """
    Formula Student Driverless Racing Environment.

    Observação (20 dims):
        [0]     vx         - velocidade longitudinal normalizada
        [1]     vy         - velocidade lateral normalizada
        [2]     omega      - yaw rate normalizado
        [3]     steering   - ângulo de viragem atual normalizado
        [4]     ax         - aceleração longitudinal normalizada
        [5:11]  blue_cones - 3 cones azuis mais próximos (x,y) no ref. do carro
        [11:17] yellow_cones - 3 cones amarelos mais próximos (x,y)
        [17]    dist_left  - distância à fronteira esquerda normalizada
        [18]    dist_right - distância à fronteira direita normalizada
        [19]    heading_err - erro de heading normalizado

    Ação (2 dims contínuas):
        [0] steering_cmd  ∈ [-1, 1]
        [1] throttle_cmd  ∈ [-1, 1] (negativo = travão)
    """

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
        dt: float = 0.02,
        action_repeat: int = 2,
        # --- FS-AI Penalty Parameters ---
        cone_penalty_reward: float = -10.0,         # reward penalty per cone hit
        cone_penalty_seconds: float = 2.0,           # FS rule: 2s per cone
        orange_cone_penalty_reward: float = -20.0,   # orange cones = worse
        doo_cone_limit: int = 10,                    # DOO after this many cones
        oc_lateral_limit: float = None,              # Off-Course lateral limit (auto)
        oc_extreme_limit: float = 5.0,               # Instant DOO if >5m off centerline
        oc_time_limit: float = 2.0,                  # OC for >2s = DOO
    ):
        super().__init__()
        self.render_mode = render_mode
        self.randomize_track = randomize_track
        self.max_episode_steps = max_episode_steps
        self.domain_randomization = domain_randomization
        self.dt = dt
        self.action_repeat = action_repeat

        # FS-AI penalty params
        self.cone_penalty_reward = cone_penalty_reward
        self.cone_penalty_seconds = cone_penalty_seconds
        self.orange_cone_penalty_reward = orange_cone_penalty_reward
        self.doo_cone_limit = doo_cone_limit
        self.oc_extreme_limit = oc_extreme_limit
        self.oc_time_limit = oc_time_limit

        # Parâmetros
        self.vehicle_params = vehicle_params or VehicleParams()
        self.track_params = track_params or TrackParams()
        self.track_seed = track_seed

        # OC lateral limit defaults to track half-width + margin
        self._oc_lateral_limit_override = oc_lateral_limit

        # Componentes
        self.car = KinematicBicycleModel(self.vehicle_params, dt)
        self.track_gen = TrackGenerator(self.track_params, track_seed)
        self.cone_sensor = ConeSensor()
        self.boundary_sensor = BoundarySensor()

        # Espaços de ação e observação
        self.obs_dim = 20
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Estado interno
        self.track_data = None
        self.current_step = 0
        self.total_progress = 0.0
        self.prev_progress = 0.0
        self.prev_cl_idx = 0
        self.laps_completed = 0
        self.episode_reward = 0.0
        self.prev_action = np.zeros(2)

        # --- Cone collision tracking ---
        self.knocked_blue = set()       # indices of knocked blue cones
        self.knocked_yellow = set()     # indices of knocked yellow cones
        self.knocked_orange = set()     # indices of knocked orange cones
        self.total_cones_hit = 0
        self.total_time_penalty = 0.0   # accumulated time penalty (seconds)

        # --- Off-course tracking ---
        self.oc_timer = 0.0             # time spent off-course consecutively

        # Renderer
        self._renderer = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reinicia o ambiente com nova pista (se randomize_track=True)."""
        super().reset(seed=seed)

        # Gerar nova pista
        if self.randomize_track or self.track_data is None:
            if seed is not None:
                self.track_gen.rng = np.random.RandomState(seed)
            self.track_data = self.track_gen.generate()

        td = self.track_data

        # Aplicar domain randomization
        start_offset = 0.0
        v_init = 0.0
        if self.domain_randomization:
            start_offset = np.random.uniform(-0.5, 0.5) * self.track_params.track_width * 0.3
            v_init = np.random.uniform(0.0, 3.0)
            self._randomize_vehicle()

        # Posição e heading iniciais
        start_pos = td['start_pos'].copy()
        start_heading = td['start_heading']

        # Offset lateral
        normal = td['normals'][0]
        start_pos += normal * start_offset

        self.car.reset(
            x=start_pos[0], y=start_pos[1],
            theta=start_heading, v=v_init
        )

        # Reset counters
        self.current_step = 0
        self.total_progress = 0.0
        self.prev_progress = 0.0
        self.prev_cl_idx = 0
        self.laps_completed = 0
        self.episode_reward = 0.0
        self.prev_action = np.zeros(2)
        self._trajectory = [(self.car.x, self.car.y)]

        # Reset cone collision state
        self.knocked_blue = set()
        self.knocked_yellow = set()
        self.knocked_orange = set()
        self.total_cones_hit = 0
        self.total_time_penalty = 0.0

        # Reset off-course timer
        self.oc_timer = 0.0

        # Compute OC lateral limit
        if self._oc_lateral_limit_override is not None:
            self.oc_lateral_limit = self._oc_lateral_limit_override
        else:
            # Default: slightly beyond the cones (track half-width + margin)
            self.oc_lateral_limit = self.track_params.track_width / 2.0 + 0.5

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Executa ação e retorna (obs, reward, terminated, truncated, info)."""
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Action repeat (simular vários sub-steps por decisão RL)
        for _ in range(self.action_repeat):
            self.car.step(action)

        self.current_step += 1
        self._trajectory.append((self.car.x, self.car.y))

        # Calcular reward
        reward, terminated = self._compute_reward(action)
        truncated = self.current_step >= self.max_episode_steps

        self.prev_action = action.copy()
        self.episode_reward += reward

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Constrói o vetor de observação de 20 dimensões."""
        td = self.track_data
        p = self.vehicle_params

        # Ego state (5 dims) - normalizado
        state = self.car.get_state()
        ego = np.array([
            state['v'] / p.max_speed,
            (state['omega'] * p.lr) / (p.max_speed + 1e-6),
            state['omega'] / (p.max_speed / p.wheelbase),
            state['steering'] / p.max_steering,
            state['ax'] / p.max_accel,
        ], dtype=np.float32)

        # Use active (non-knocked) cones for observations
        active_blue = self._get_active_cones('blue')
        active_yellow = self._get_active_cones('yellow')

        # Cone observations (12 dims)
        blue_obs, yellow_obs, _ = self.cone_sensor.get_observations(
            self.car.x, self.car.y, self.car.theta,
            active_blue, active_yellow,
            add_noise=self.domain_randomization
        )
        cone_obs = np.concatenate([
            blue_obs.flatten() / self.cone_sensor.max_range,
            yellow_obs.flatten() / self.cone_sensor.max_range,
        ]).astype(np.float32)

        # Boundary info (3 dims)
        boundary = self.boundary_sensor.get_boundary_info(
            self.car.x, self.car.y, self.car.theta,
            td['centerline'], td['tangents'], td['normals'],
            self.track_params.track_width / 2.0
        )

        obs = np.concatenate([ego, cone_obs, boundary])
        return np.clip(obs, -2.0, 2.0).astype(np.float32)

    def _get_active_cones(self, color: str) -> np.ndarray:
        """Retorna apenas os cones que NÃO foram derrubados."""
        td = self.track_data
        if color == 'blue':
            all_cones = td['blue_cones']
            knocked = self.knocked_blue
        elif color == 'yellow':
            all_cones = td['yellow_cones']
            knocked = self.knocked_yellow
        else:
            all_cones = td['orange_cones']
            knocked = self.knocked_orange

        if len(knocked) == 0:
            return all_cones

        mask = np.ones(len(all_cones), dtype=bool)
        for idx in knocked:
            if idx < len(all_cones):
                mask[idx] = False

        active = all_cones[mask]
        # Return at least an empty array with right shape
        if len(active) == 0:
            return np.empty((0, 2), dtype=all_cones.dtype)
        return active

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool]:
        """
        Calcula a recompensa baseada em progresso, velocidade, suavidade e
        penalizações FS-AI realistas por cones derrubados e off-course.
        """
        td = self.track_data
        cl = td['centerline']
        n_cl = len(cl)

        # Encontrar posição na centerline
        diffs = cl - np.array([self.car.x, self.car.y])
        dists_sq = np.sum(diffs**2, axis=1)
        cl_idx = int(np.argmin(dists_sq))
        lateral_dist = np.sqrt(dists_sq[cl_idx])

        # ---- PROGRESSO ----
        diff_idx = (cl_idx - self.prev_cl_idx) % n_cl
        if diff_idx > n_cl // 2:
            diff_idx -= n_cl

        progress = 0.0
        if diff_idx > 0:
            for i in range(diff_idx):
                idx_from = (self.prev_cl_idx + i) % n_cl
                idx_to = (self.prev_cl_idx + i + 1) % n_cl
                progress += np.sqrt(np.sum(
                    (cl[idx_to] - cl[idx_from])**2
                ))
        elif diff_idx < 0:
            progress = diff_idx * 0.5

        self.total_progress += max(0, progress)

        # Verificar volta completa
        if (self.total_progress > td['track_length'] * 0.9 and
                cl_idx < n_cl * 0.1 and self.prev_cl_idx > n_cl * 0.8):
            self.laps_completed += 1
            self.total_progress = 0.0

        self.prev_cl_idx = cl_idx

        # ---- REWARD COMPONENTS ----

        # 1. Progresso (sinal dominante)
        r_progress = progress * 1.0

        # 2. Velocidade na direção certa
        heading_error = abs(self._get_heading_error(cl_idx))
        speed_reward = (self.car.v / self.vehicle_params.max_speed) * np.cos(heading_error)
        r_speed = speed_reward * 0.1

        # 3. Penalidade de suavidade (steering jerk)
        steering_change = abs(action[0] - self.prev_action[0])
        r_smooth = -steering_change * 0.3

        # 4. Penalidade lateral (quadrática)
        track_hw = self.track_params.track_width / 2.0
        lateral_ratio = lateral_dist / track_hw
        r_lateral = -(lateral_ratio ** 2) * 0.2

        # 5. Time penalty (encorajar velocidade)
        r_time = -0.01

        # Reward total (base)
        reward = r_progress + r_speed + r_smooth + r_lateral + r_time

        # ---- CONE COLLISION (FS-AI RULES) ----
        # Cones derrubados = penalidade, NÃO terminação
        cone_hit_reward = self._process_cone_collisions()
        reward += cone_hit_reward

        # ---- OFF-COURSE HANDLING (FS-AI RULES) ----
        terminated = False
        step_time = self.dt * self.action_repeat

        is_off_course = lateral_dist > self.oc_lateral_limit

        if is_off_course:
            self.oc_timer += step_time
            # Gradual penalty while off-course (stronger the further out)
            oc_ratio = lateral_dist / self.oc_lateral_limit
            reward -= 0.5 * (oc_ratio ** 2)
        else:
            self.oc_timer = 0.0  # reset if back on track

        # --- CONDIÇÕES TERMINAIS (DOO - Did Not Operate) ---

        # 1. DOO: Demasiados cones derrubados
        if self.total_cones_hit >= self.doo_cone_limit:
            reward -= 50.0
            terminated = True

        # 2. DOO: Off-course extremo (>5m do centerline)
        if lateral_dist > self.oc_extreme_limit:
            reward -= 50.0
            terminated = True

        # 3. DOO: Off-course prolongado (>2s fora dos limites)
        if self.oc_timer >= self.oc_time_limit:
            reward -= 30.0
            terminated = True

        # 4. Volta completa - bónus (penalidade de cones descontada)
        if self.laps_completed > 0:
            lap_bonus = 200.0 - (self.total_cones_hit * 5.0)
            reward += max(lap_bonus, 50.0)
            terminated = True

        # 5. Contramão prolongada
        if diff_idx < -5:
            reward -= 20.0
            terminated = True

        # 6. Velocidade zero por muito tempo (estagnação)
        if self.car.v < 0.3 and self.current_step > 100:
            reward -= 1.0

        self.prev_progress = self.total_progress

        return float(reward), terminated

    def _process_cone_collisions(self) -> float:
        """
        Verifica colisões com cones e aplica penalizações FS-AI.
        Cones derrubados são marcados e removidos da pista ativa.
        Retorna a penalização de reward para este step.
        """
        td = self.track_data
        car_pos = np.array([self.car.x, self.car.y])
        cone_radius = 0.15  # raio do cone (30cm altura, ~15cm raio base)
        car_radius = max(self.vehicle_params.width, self.vehicle_params.length) / 2.0

        collision_dist = car_radius + cone_radius
        penalty = 0.0

        # Check blue cones
        for i, cone in enumerate(td['blue_cones']):
            if i not in self.knocked_blue:
                dist = np.sqrt(np.sum((cone - car_pos) ** 2))
                if dist < collision_dist:
                    self.knocked_blue.add(i)
                    self.total_cones_hit += 1
                    self.total_time_penalty += self.cone_penalty_seconds
                    penalty += self.cone_penalty_reward

        # Check yellow cones
        for i, cone in enumerate(td['yellow_cones']):
            if i not in self.knocked_yellow:
                dist = np.sqrt(np.sum((cone - car_pos) ** 2))
                if dist < collision_dist:
                    self.knocked_yellow.add(i)
                    self.total_cones_hit += 1
                    self.total_time_penalty += self.cone_penalty_seconds
                    penalty += self.cone_penalty_reward

        # Check orange cones (start/finish - penalidade agravada)
        for i, cone in enumerate(td['orange_cones']):
            if i not in self.knocked_orange:
                dist = np.sqrt(np.sum((cone - car_pos) ** 2))
                if dist < collision_dist:
                    self.knocked_orange.add(i)
                    self.total_cones_hit += 1
                    self.total_time_penalty += self.cone_penalty_seconds * 2
                    penalty += self.orange_cone_penalty_reward

        return penalty

    def _get_heading_error(self, cl_idx: int) -> float:
        """Calcula erro de heading relativo à tangente da pista."""
        tangent = self.track_data['tangents'][cl_idx]
        track_heading = np.arctan2(tangent[1], tangent[0])
        error = self.car.theta - track_heading
        return (error + np.pi) % (2 * np.pi) - np.pi

    def _randomize_vehicle(self):
        """Aplica domain randomization aos parâmetros do veículo."""
        p = self.vehicle_params
        p.mass = 250.0 * np.random.uniform(0.85, 1.15)
        p.mu = 1.5 * np.random.uniform(0.80, 1.20)
        p.drag_coeff = 0.02 * np.random.uniform(0.90, 1.10)

    def _get_info(self) -> dict:
        """Retorna informações adicionais do episódio."""
        return {
            'speed': self.car.v,
            'speed_kmh': self.car.v * 3.6,
            'total_progress': self.total_progress,
            'laps_completed': self.laps_completed,
            'episode_reward': self.episode_reward,
            'step': self.current_step,
            'x': self.car.x,
            'y': self.car.y,
            'theta': self.car.theta,
            'steering': self.car.steering,
            # --- FS-AI penalty info ---
            'cones_hit': self.total_cones_hit,
            'cones_hit_blue': len(self.knocked_blue),
            'cones_hit_yellow': len(self.knocked_yellow),
            'cones_hit_orange': len(self.knocked_orange),
            'time_penalty': self.total_time_penalty,
            'off_course_timer': self.oc_timer,
            'doo_limit': self.doo_cone_limit,
        }

    def render(self):
        """Renderiza o ambiente."""
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from .renderer import FSRenderer
            self._renderer = FSRenderer(self)

        return self._renderer.render()

    def close(self):
        """Fecha recursos."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Registar no Gymnasium
gym.register(
    id='FSRacing-v0',
    entry_point='env.racing_env:FSRacingEnv',
    max_episode_steps=5000,
)
