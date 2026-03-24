"""
Ambiente Gymnasium para corrida Formula Student com cones.
Observação: estado do ego + cones + boundary info (~20 dims)
Ação: [steering, throttle] contínuos em [-1, 1]
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
    ):
        super().__init__()
        self.render_mode = render_mode
        self.randomize_track = randomize_track
        self.max_episode_steps = max_episode_steps
        self.domain_randomization = domain_randomization
        self.dt = dt
        self.action_repeat = action_repeat

        # Parâmetros
        self.vehicle_params = vehicle_params or VehicleParams()
        self.track_params = track_params or TrackParams()
        self.track_seed = track_seed

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
            # Posição inicial com offset aleatório
            start_offset = np.random.uniform(-0.5, 0.5) * self.track_params.track_width * 0.3
            v_init = np.random.uniform(0.0, 3.0)  # velocidade inicial aleatória
            # Randomizar parâmetros do veículo
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
            state['v'] / p.max_speed,                     # vx normalizado
            (state['omega'] * p.lr) / (p.max_speed + 1e-6),  # vy approx normalizado
            state['omega'] / (p.max_speed / p.wheelbase),  # omega normalizado
            state['steering'] / p.max_steering,             # steering normalizado
            state['ax'] / p.max_accel,                      # ax normalizado
        ], dtype=np.float32)

        # Cone observations (12 dims)
        blue_obs, yellow_obs, _ = self.cone_sensor.get_observations(
            self.car.x, self.car.y, self.car.theta,
            td['blue_cones'], td['yellow_cones'],
            add_noise=self.domain_randomization
        )
        # Normalizar pelo alcance máximo
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

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool]:
        """Calcula a recompensa baseada em progresso, velocidade e suavidade."""
        td = self.track_data
        cl = td['centerline']
        n_cl = len(cl)

        # Encontrar posição na centerline
        diffs = cl - np.array([self.car.x, self.car.y])
        dists_sq = np.sum(diffs**2, axis=1)
        cl_idx = int(np.argmin(dists_sq))
        lateral_dist = np.sqrt(dists_sq[cl_idx])

        # ---- PROGRESSO ----
        # Diferença de índice na centerline (com wrapping)
        diff_idx = (cl_idx - self.prev_cl_idx) % n_cl
        if diff_idx > n_cl // 2:
            diff_idx -= n_cl  # contramão

        # Converter para distância real
        progress = 0.0
        if diff_idx > 0:
            for i in range(diff_idx):
                idx_from = (self.prev_cl_idx + i) % n_cl
                idx_to = (self.prev_cl_idx + i + 1) % n_cl
                progress += np.sqrt(np.sum(
                    (cl[idx_to] - cl[idx_from])**2
                ))
        elif diff_idx < 0:
            progress = diff_idx * 0.5  # penalidade por contramão

        self.total_progress += max(0, progress)

        # Verificar volta completa
        if (self.total_progress > td['track_length'] * 0.9 and
                cl_idx < n_cl * 0.1 and self.prev_cl_idx > n_cl * 0.8):
            self.laps_completed += 1
            self.total_progress = 0.0  # reset para nova volta

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

        # Reward total
        reward = r_progress + r_speed + r_smooth + r_lateral + r_time

        # ---- CONDIÇÕES TERMINAIS ----
        terminated = False

        # Colisão com cone
        if self._check_cone_collision():
            reward = -50.0
            terminated = True

        # Fora da pista
        if lateral_dist > track_hw * 1.3:
            reward = -100.0
            terminated = True

        # Volta completa - bónus
        if self.laps_completed > 0:
            reward += 200.0
            terminated = True

        # Velocidade zero por muito tempo (estagnação)
        if self.car.v < 0.3 and self.current_step > 100:
            reward -= 1.0

        self.prev_progress = self.total_progress

        return float(reward), terminated

    def _check_cone_collision(self) -> bool:
        """Verifica se o carro colidiu com algum cone."""
        td = self.track_data
        car_pos = np.array([self.car.x, self.car.y])
        cone_radius = 0.15  # raio do cone (m)
        car_radius = max(self.vehicle_params.width, self.vehicle_params.length) / 2

        all_cones = np.vstack([
            td['blue_cones'], td['yellow_cones'], td['orange_cones']
        ])

        dists = np.sqrt(np.sum((all_cones - car_pos)**2, axis=1))
        return bool(np.any(dists < (car_radius + cone_radius)))

    def _get_heading_error(self, cl_idx: int) -> float:
        """Calcula erro de heading relativo à tangente da pista."""
        tangent = self.track_data['tangents'][cl_idx]
        track_heading = np.arctan2(tangent[1], tangent[0])
        error = self.car.theta - track_heading
        return (error + np.pi) % (2 * np.pi) - np.pi

    def _randomize_vehicle(self):
        """Aplica domain randomization aos parâmetros do veículo."""
        p = self.vehicle_params
        # ±15% na massa
        p.mass = 250.0 * np.random.uniform(0.85, 1.15)
        # ±20% no atrito
        p.mu = 1.5 * np.random.uniform(0.80, 1.20)
        # ±10% no arrasto
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
