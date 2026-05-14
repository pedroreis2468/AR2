"""
PacSimEnv — ponte Gymnasium ↔ ROS 2 / PacSim para sim-to-sim transfer.

A interface (observação 24-dim, ação 2-dim) replica exactamente o
FSRacingEnv(use_orange_cones=True) para que uma política treinada no
ambiente 2D possa ser avaliada directamente no simulador 3D.

Pré-condições (em terminais separados):
    conda deactivate
    source ~/pacsim_ws/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    ros2 launch pacsim example.launch.py

Tópicos:
    Subscribe
        /pacsim/velocity            geometry_msgs/TwistWithCovarianceStamped
        /pacsim/imu/cog_imu         sensor_msgs/Imu
        /pacsim/steeringFront       pacsim/StampedScalar
        /pacsim/perception/livox_front/landmarks
                                    pacsim/PerceptionDetections
    Publish
        /pacsim/steering_setpoint   pacsim/StampedScalar
        /pacsim/wheelspeed_setpoints
                                    pacsim/Wheels

Sincronização step() ↔ ROS:
    O nó rclpy gira numa thread daemon (MultiThreadedExecutor). O step()
    do Gymnasium publica a acção, regista o stamp ROS desse instante, e
    bloqueia até chegar uma perceção com stamp > stamp da acção (ou até
    expirar `step_timeout`). Isto torna o passo do Gym indexado ao clock
    do ROS — funciona tanto com wall-clock como com /clock (use_sim_time),
    e elimina o risco de obter uma observação "velha" capturada antes do
    comando ser aplicado. Ver bloco docstring no fim do ficheiro.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.time import Time as RclTime

    from geometry_msgs.msg import TwistWithCovarianceStamped
    from sensor_msgs.msg import Imu
    from pacsim.msg import PerceptionDetections, StampedScalar, Wheels

    _ROS_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # ImportError ou ModuleNotFoundError para pacsim.msg
    _ROS_IMPORT_ERROR = str(exc)
    # Stubs para o módulo permanecer importável (e.g. em CI ou em máquinas
    # sem ROS 2 instalado). Qualquer tentativa de instanciar PacSimEnv
    # levanta RuntimeError com instruções, antes de tocar nestes símbolos.
    Node = object  # type: ignore[assignment,misc]
    QoSProfile = ReliabilityPolicy = HistoryPolicy = None  # type: ignore[assignment]
    MultiThreadedExecutor = RclTime = None  # type: ignore[assignment]
    TwistWithCovarianceStamped = Imu = None  # type: ignore[assignment]
    PerceptionDetections = StampedScalar = Wheels = None  # type: ignore[assignment]


# ── Parâmetros do veículo (paridade com VehicleParams do FSRacingEnv) ──────
_MAX_SPEED = 28.0
_MAX_STEERING = 0.4
_MAX_ACCEL = 12.0
_MU = 1.5
_LR = 0.775
_WHEELBASE = 1.55
_MAX_LAT_ACCEL = _MU * 9.81
_MAX_OMEGA = _MAX_SPEED / _WHEELBASE

# ── Sensor / normalização (paridade com ConeSensor do FSRacingEnv) ─────────
_WHEEL_RADIUS = 0.206
_CONE_MAX_RANGE = 15.0
_N_CONES_PER_COLOR = 3

# ── Classes da Landmark.msg do PacSim ──────────────────────────────────────
_CLS_BLUE = 2
_CLS_YELLOW = 3
_CLS_ORANGE = 4
_CLS_BIGORANGE = 5


@dataclass
class _Snapshot:
    """Cópia imutável do estado do nó num dado instante."""
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    steering: float = 0.0
    cones: tuple = ()                  # tuplo de (x, y, cls)
    perception_stamp_ns: int = 0       # ROS stamp da última perceção, em ns


class _PacSimBridgeNode(Node):
    """Nó rclpy: lê sensores, publica comandos, expõe snapshot thread-safe."""

    def __init__(self, perception_topic: str):
        super().__init__('ar2_pacsim_bridge')
        self._lock = threading.Lock()
        self._state = _Snapshot()
        self._perception_event = threading.Event()

        # QoS para sensores: BEST_EFFORT é o que a maioria dos drivers de
        # LiDAR/IMU usa. KEEP_LAST(1) garante que ficamos sempre com a leitura
        # mais recente em vez de drenar uma fila acumulada.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            TwistWithCovarianceStamped, '/pacsim/velocity',
            self._on_velocity, sensor_qos)
        self.create_subscription(
            Imu, '/pacsim/imu/cog_imu',
            self._on_imu, sensor_qos)
        self.create_subscription(
            StampedScalar, '/pacsim/steeringFront',
            self._on_steering, sensor_qos)
        self.create_subscription(
            PerceptionDetections, perception_topic,
            self._on_perception, sensor_qos)

        # Publishers: comandos de controlo são RELIABLE + KEEP_LAST(1).
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub_steer = self.create_publisher(
            StampedScalar, '/pacsim/steering_setpoint', cmd_qos)
        self._pub_wheels = self.create_publisher(
            Wheels, '/pacsim/wheelspeed_setpoints', cmd_qos)

    # ── Callbacks ───────────────────────────────────────────────────────

    def _on_velocity(self, msg):
        with self._lock:
            self._state.vx = msg.twist.twist.linear.x
            self._state.vy = msg.twist.twist.linear.y
            self._state.omega = msg.twist.twist.angular.z

    def _on_imu(self, msg):
        with self._lock:
            self._state.ax = msg.linear_acceleration.x
            self._state.ay = msg.linear_acceleration.y

    def _on_steering(self, msg):
        with self._lock:
            self._state.steering = float(msg.value)

    def _on_perception(self, msg):
        cones = []
        for lm in msg.detections:
            probs = lm.class_probabilities
            # probs vem como numpy array de tamanho fixo — len() evita o
            # `ValueError: truth value of an array … is ambiguous`.
            if len(probs) == 0:
                continue
            cls = int(np.argmax(probs))
            cones.append((
                float(lm.pose.pose.position.x),
                float(lm.pose.pose.position.y),
                cls,
            ))
        stamp = msg.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        with self._lock:
            self._state.cones = tuple(cones)
            self._state.perception_stamp_ns = stamp_ns
        self._perception_event.set()

    # ── API usada pelo Env ─────────────────────────────────────────────

    def snapshot(self) -> _Snapshot:
        with self._lock:
            s = self._state
            return _Snapshot(
                vx=s.vx, vy=s.vy, omega=s.omega,
                ax=s.ax, ay=s.ay, steering=s.steering,
                cones=s.cones,
                perception_stamp_ns=s.perception_stamp_ns,
            )

    def now_ns(self) -> int:
        t: RclTime = self.get_clock().now()
        return int(t.nanoseconds)

    def wait_next_perception(self, last_stamp_ns: int, timeout_s: float) -> bool:
        """
        Bloqueia até chegar uma perceção com stamp > last_stamp_ns.

        Comparamos *stamps de mensagens entre si* (monotónico, vem todo da
        mesma fonte de tempo do PacSim), em vez de comparar contra
        `node.now_ns()`. Assim funciona quer o publisher use wall-clock,
        quer use sim-time (`use_sim_time:=true` com /clock).
        """
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                if self._state.perception_stamp_ns > last_stamp_ns:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            self._perception_event.clear()
            self._perception_event.wait(timeout=min(remaining, 0.05))

    def publish_action(self, steering_norm: float, throttle_norm: float):
        stamp = self.get_clock().now().to_msg()

        steer = StampedScalar()
        steer.stamp = stamp
        steer.value = float(np.clip(steering_norm, -1.0, 1.0) * _MAX_STEERING)
        self._pub_steer.publish(steer)

        # throttle_norm ∈ [-1, 1]. Negativo = travão. PacSim não tem tópico
        # de brake dedicado neste setup; mapeamos travão para velocidade-alvo
        # nula (coast) — o veículo desacelera naturalmente.
        target_speed = max(0.0, float(throttle_norm)) * _MAX_SPEED
        wheel_rate = target_speed / _WHEEL_RADIUS

        wheels = Wheels()
        wheels.stamp = stamp
        wheels.fl = wheel_rate
        wheels.fr = wheel_rate
        wheels.rl = wheel_rate
        wheels.rr = wheel_rate
        self._pub_wheels.publish(wheels)


class PacSimEnv(gym.Env):
    """
    Gymnasium Env que liga uma política AR2 ao PacSim via ROS 2.

    Args:
        perception_topic: tópico de perceção a subscrever.
        step_timeout: tempo máximo (s) à espera de uma perceção fresca.
        startup_timeout: tempo máximo (s) à espera da primeira perceção.
        use_orange_cones: 24-dim (True) ou 18-dim (False).
        max_episode_steps: trunca o episódio quando atingido.
    """

    metadata = {'render_modes': [], 'render_fps': 0}

    def __init__(
        self,
        perception_topic: str = '/pacsim/perception/livox_front/landmarks',
        step_timeout: float = 1.0,
        startup_timeout: float = 15.0,
        use_orange_cones: bool = True,
        max_episode_steps: int = 5000,
    ):
        if _ROS_IMPORT_ERROR is not None:
            raise RuntimeError(
                f"Falha a importar rclpy/pacsim.msg: {_ROS_IMPORT_ERROR}\n"
                f"Antes de instanciar PacSimEnv corre:\n"
                f"  conda deactivate\n"
                f"  source ~/pacsim_ws/install/setup.bash\n"
                f"  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
            )
        super().__init__()

        self.step_timeout = float(step_timeout)
        self.use_orange_cones = bool(use_orange_cones)
        self.max_episode_steps = int(max_episode_steps)

        obs_dim = 24 if self.use_orange_cones else 18
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -2.0, 2.0, shape=(obs_dim,), dtype=np.float32)

        if not rclpy.ok():
            rclpy.init()

        self._node = _PacSimBridgeNode(perception_topic)
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self._step_count = 0
        print('[PacSimEnv] À espera da primeira perceção do PacSim…')
        got = self._node.wait_next_perception(
            last_stamp_ns=0, timeout_s=startup_timeout)
        if not got:
            self.close()
            raise TimeoutError(
                f'PacSim não publicou em {startup_timeout:.0f}s no tópico '
                f'{perception_topic}. Confirma o launch do simulador.'
            )
        print('[PacSimEnv] Ligação estabelecida.')

    # ── Gymnasium API ──────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        last_stamp = self._node.snapshot().perception_stamp_ns
        # Comando neutro para "parar" o carro entre episódios.
        self._node.publish_action(0.0, 0.0)
        self._node.wait_next_perception(
            last_stamp_ns=last_stamp, timeout_s=self.step_timeout)
        return self._build_observation(), self._build_info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        last_stamp = self._node.snapshot().perception_stamp_ns
        self._node.publish_action(float(action[0]), float(action[1]))

        fresh = self._node.wait_next_perception(
            last_stamp_ns=last_stamp, timeout_s=self.step_timeout)

        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        info = self._build_info()
        info['stale_perception'] = (not fresh)
        # Reward = 0: sim-to-sim transfer avalia uma política já treinada.
        return self._build_observation(), 0.0, False, truncated, info

    def close(self):
        try:
            self._node.publish_action(0.0, 0.0)
        except Exception:
            pass
        try:
            self._executor.shutdown(timeout_sec=0.5)
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()

    # ── Observação (paridade FSRacingEnv._get_obs com use_orange=True) ──

    def _build_observation(self) -> np.ndarray:
        s = self._node.snapshot()

        # Nota: o índice 1 usa o proxy (ω·lr)/v_max — exactamente como o
        # FSRacingEnv. Apesar de termos vy "real" no twist do PacSim, usar o
        # mesmo proxy mantém a distribuição de observações alinhada com o
        # ambiente onde a política foi treinada.
        ego = np.array([
            s.vx / _MAX_SPEED,
            (s.omega * _LR) / (_MAX_SPEED + 1e-6),
            s.omega / _MAX_OMEGA,
            s.steering / _MAX_STEERING,
            s.ax / _MAX_ACCEL,
            s.ay / _MAX_LAT_ACCEL,
        ], dtype=np.float32)

        blues, yellows, oranges = [], [], []
        for x, y, cls in s.cones:
            if (x * x + y * y) > (_CONE_MAX_RANGE * _CONE_MAX_RANGE):
                continue
            if cls == _CLS_BLUE:
                blues.append((x, y))
            elif cls == _CLS_YELLOW:
                yellows.append((x, y))
            elif cls in (_CLS_ORANGE, _CLS_BIGORANGE):
                oranges.append((x, y))

        b = _closest_n(blues) / _CONE_MAX_RANGE
        y = _closest_n(yellows) / _CONE_MAX_RANGE
        if self.use_orange_cones:
            o = _closest_n(oranges) / _CONE_MAX_RANGE
            cones = np.concatenate([b.ravel(), y.ravel(), o.ravel()])
        else:
            cones = np.concatenate([b.ravel(), y.ravel()])

        obs = np.concatenate([ego, cones.astype(np.float32)])
        return np.clip(obs, -2.0, 2.0).astype(np.float32)

    def _build_info(self) -> dict:
        s = self._node.snapshot()
        return {
            'speed': s.vx,
            'speed_kmh': s.vx * 3.6,
            'steering': s.steering,
            'step': self._step_count,
            'n_cones_visible': len(s.cones),
        }


def _closest_n(cones: list, n: int = _N_CONES_PER_COLOR) -> np.ndarray:
    """Devolve os `n` cones mais próximos como array (n, 2); zeros em falta."""
    out = np.zeros((n, 2), dtype=np.float32)
    if not cones:
        return out
    arr = np.asarray(cones, dtype=np.float32)
    d = arr[:, 0] ** 2 + arr[:, 1] ** 2
    idx = np.argsort(d)[:n]
    out[: len(idx)] = arr[idx]
    return out


gym.register(
    id='PacSim-v0',
    entry_point='env.pacsim_env:PacSimEnv',
    max_episode_steps=5000,
)
