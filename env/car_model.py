"""
Modelo cinemático de bicicleta para carro Formula Student.
Referência: Rajamani (2012) - Vehicle Dynamics and Control, Cap. 2
Parâmetros calibrados para um FS típico (~250kg, wheelbase 1.55m).
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class VehicleParams:
    """Parâmetros do veículo Formula Student."""
    mass: float = 250.0           # kg
    wheelbase: float = 1.55       # m (distância entre eixos)
    lr: float = 0.775             # m (CG ao eixo traseiro)
    lf: float = 0.775             # m (CG ao eixo dianteiro)
    max_steering: float = 0.4     # rad (~23°)
    max_speed: float = 28.0       # m/s (~100 km/h)
    max_accel: float = 12.0       # m/s² (~1.2g)
    max_brake: float = 18.0       # m/s² (~1.8g)
    max_steering_rate: float = 1.5  # rad/s (taxa máxima de viragem)
    drag_coeff: float = 0.02      # coeficiente de arrasto simplificado
    mu: float = 1.5               # coeficiente de atrito (slicks em seco)
    width: float = 1.4            # m (largura do carro para colisão)
    length: float = 3.0           # m (comprimento do carro para render)


class KinematicBicycleModel:
    """
    Modelo cinemático de bicicleta com referência no CG.
    Estado: [x, y, theta, v, steering_angle]
    Ação: [steering_cmd ∈ [-1,1], throttle_cmd ∈ [-1,1]]
    """

    def __init__(self, params: VehicleParams = None, dt: float = 0.02):
        self.params = params or VehicleParams()
        self.dt = dt
        self.reset()

    def reset(self, x: float = 0.0, y: float = 0.0,
              theta: float = 0.0, v: float = 0.0):
        """Reinicia o estado do veículo."""
        self.x = x
        self.y = y
        self.theta = theta          # heading (rad)
        self.v = v                   # velocidade longitudinal (m/s)
        self.steering = 0.0          # ângulo de viragem atual (rad)
        self.omega = 0.0             # yaw rate (rad/s)
        self.ax = 0.0                # aceleração longitudinal
        self.ay = 0.0                # aceleração lateral
        self.prev_steering = 0.0

    def step(self, action: np.ndarray) -> dict:
        """
        Avança um passo dt.
        action[0]: steering_cmd ∈ [-1, 1] → ângulo de viragem desejado
        action[1]: throttle_cmd ∈ [-1, 1] → negativo = travão, positivo = aceleração
        Retorna dict com info do estado.
        """
        p = self.params
        steer_cmd = np.clip(action[0], -1.0, 1.0)
        throttle_cmd = np.clip(action[1], -1.0, 1.0)

        # --- Steering com rate limiting ---
        target_steering = steer_cmd * p.max_steering
        delta_steer = target_steering - self.steering
        max_delta = p.max_steering_rate * self.dt
        delta_steer = np.clip(delta_steer, -max_delta, max_delta)
        self.prev_steering = self.steering
        self.steering = np.clip(
            self.steering + delta_steer,
            -p.max_steering, p.max_steering
        )

        # --- Aceleração / Travagem ---
        if throttle_cmd >= 0:
            accel = throttle_cmd * p.max_accel
        else:
            accel = throttle_cmd * p.max_brake

        # Arrasto aerodinâmico simplificado
        drag = -p.drag_coeff * self.v * abs(self.v)
        self.ax = accel + drag

        # --- Friction circle constraint ---
        # a_lat ≈ v² * tan(δ) / L para modelo cinemático
        if abs(self.v) > 0.5:
            a_lat_approx = (self.v ** 2) * np.tan(self.steering) / p.wheelbase
        else:
            a_lat_approx = 0.0

        max_total_accel = p.mu * 9.81
        if abs(a_lat_approx) > max_total_accel:
            # Reduzir velocidade se ultrapassar atrito lateral
            a_lat_approx = np.sign(a_lat_approx) * max_total_accel

        remaining_accel = np.sqrt(
            max(0, max_total_accel**2 - a_lat_approx**2)
        )
        self.ax = np.clip(self.ax, -remaining_accel, remaining_accel)
        self.ay = a_lat_approx

        # --- Atualização cinemática (referência no CG) ---
        beta = np.arctan2(p.lr * np.tan(self.steering), p.wheelbase)

        self.x += self.v * np.cos(self.theta + beta) * self.dt
        self.y += self.v * np.sin(self.theta + beta) * self.dt
        self.omega = (self.v / p.lr) * np.sin(beta) if p.lr > 0 else 0.0
        self.theta += self.omega * self.dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi  # normalizar
        self.v += self.ax * self.dt
        self.v = np.clip(self.v, 0.0, p.max_speed)

        return self.get_state()

    def get_state(self) -> dict:
        """Retorna estado completo do veículo."""
        return {
            'x': self.x,
            'y': self.y,
            'theta': self.theta,
            'v': self.v,
            'steering': self.steering,
            'omega': self.omega,
            'ax': self.ax,
            'ay': self.ay,
            'vx': self.v * np.cos(self.theta),
            'vy': self.v * np.sin(self.theta),
        }

    def get_corners(self) -> np.ndarray:
        """Retorna os 4 cantos do carro (para colisão)."""
        p = self.params
        hw = p.width / 2
        hl = p.length / 2
        corners_local = np.array([
            [-hl, -hw], [-hl, hw], [hl, hw], [hl, -hw]
        ])
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        return (corners_local @ rot.T) + np.array([self.x, self.y])
