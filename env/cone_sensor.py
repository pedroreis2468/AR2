"""
Simulação do sensor de cones - imita a pipeline de perceção de um FS car.
Deteta os cones mais próximos no referencial do carro, com ruído gaussiano
para simular imperfeições do YOLO + depth camera.
"""
import numpy as np
from typing import Tuple


class ConeSensor:
    """
    Simula a perceção de cones no referencial do carro.
    Equivalente ao output de YOLO + stereo depth num FS real.
    """

    def __init__(
        self,
        max_range: float = 15.0,      # alcance máximo de deteção (m)
        fov: float = np.pi,            # campo de visão (rad) - 180°
        n_closest_per_side: int = 3,   # nº de cones mais próximos por lado
        noise_range: float = 0.2,      # ruído na distância (m)
        noise_bearing: float = 0.007,  # ruído no ângulo (rad)
        detection_prob: float = 0.95,  # probabilidade de detetar um cone
    ):
        self.max_range = max_range
        self.fov = fov
        self.n_closest = n_closest_per_side
        self.noise_range = noise_range
        self.noise_bearing = noise_bearing
        self.detection_prob = detection_prob

    def get_observations(
        self,
        car_x: float, car_y: float, car_theta: float,
        blue_cones: np.ndarray, yellow_cones: np.ndarray,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retorna observações dos cones no referencial do carro.

        Returns:
            blue_obs: (n_closest, 2) - posições dos cones azuis mais próximos [x_car, y_car]
            yellow_obs: (n_closest, 2) - posições dos cones amarelos mais próximos
            visible_info: dict com informação de debug
        """
        blue_obs = self._detect_cones(
            car_x, car_y, car_theta, blue_cones, add_noise
        )
        yellow_obs = self._detect_cones(
            car_x, car_y, car_theta, yellow_cones, add_noise
        )

        visible_info = {
            'n_blue_visible': np.sum(np.any(blue_obs != 0, axis=1)),
            'n_yellow_visible': np.sum(np.any(yellow_obs != 0, axis=1)),
        }

        return blue_obs, yellow_obs, visible_info

    def _detect_cones(
        self,
        car_x: float, car_y: float, car_theta: float,
        cones: np.ndarray, add_noise: bool
    ) -> np.ndarray:
        """Deteta os n_closest cones mais próximos no referencial do carro."""
        if len(cones) == 0:
            return np.zeros((self.n_closest, 2), dtype=np.float32)

        # Transformar cones para referencial do carro
        dx = cones[:, 0] - car_x
        dy = cones[:, 1] - car_y

        cos_t = np.cos(-car_theta)
        sin_t = np.sin(-car_theta)
        x_car = dx * cos_t - dy * sin_t
        y_car = dx * sin_t + dy * cos_t

        # Filtrar por alcance e FOV
        distances = np.sqrt(x_car**2 + y_car**2)
        bearings = np.arctan2(y_car, x_car)

        in_range = distances <= self.max_range
        in_fov = np.abs(bearings) <= self.fov / 2
        visible = in_range & in_fov

        # Simular falhas de deteção
        if add_noise:
            detection_mask = np.random.random(len(cones)) < self.detection_prob
            visible = visible & detection_mask

        if not visible.any():
            return np.zeros((self.n_closest, 2), dtype=np.float32)

        # Selecionar os mais próximos
        visible_dists = np.where(visible, distances, np.inf)
        closest_idx = np.argsort(visible_dists)[:self.n_closest]

        result = np.zeros((self.n_closest, 2), dtype=np.float32)
        for i, idx in enumerate(closest_idx):
            if not visible[idx]:
                break
            xc = x_car[idx]
            yc = y_car[idx]

            # Adicionar ruído
            if add_noise:
                dist = distances[idx]
                bearing = bearings[idx]
                dist += np.random.normal(0, self.noise_range)
                bearing += np.random.normal(0, self.noise_bearing)
                xc = dist * np.cos(bearing)
                yc = dist * np.sin(bearing)

            result[i] = [xc, yc]

        return result

    def get_observation_size(self) -> int:
        """Tamanho total do vetor de observação de cones."""
        return self.n_closest * 2 * 2  # 2 lados × n_closest × 2 coords


class BoundarySensor:
    """
    Calcula distância perpendicular às fronteiras e heading error.
    Complementa o ConeSensor com informação global de posicionamento.
    """

    @staticmethod
    def get_boundary_info(
        car_x: float, car_y: float, car_theta: float,
        centerline: np.ndarray, tangents: np.ndarray,
        normals: np.ndarray, track_half_width: float
    ) -> np.ndarray:
        """
        Retorna [dist_left, dist_right, heading_error] normalizados.
        """
        # Encontrar ponto mais próximo na centerline
        diffs = centerline - np.array([car_x, car_y])
        dists_sq = np.sum(diffs**2, axis=1)
        idx = int(np.argmin(dists_sq))

        # Distância lateral (positiva = à esquerda da centerline)
        to_car = np.array([car_x - centerline[idx, 0],
                           car_y - centerline[idx, 1]])
        lateral_dist = np.dot(to_car, normals[idx])

        dist_left = track_half_width - lateral_dist
        dist_right = track_half_width + lateral_dist

        # Heading error relativo à tangente local
        track_heading = np.arctan2(tangents[idx, 1], tangents[idx, 0])
        heading_error = car_theta - track_heading
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # Normalizar
        return np.array([
            np.clip(dist_left / track_half_width, -1.0, 2.0),
            np.clip(dist_right / track_half_width, -1.0, 2.0),
            heading_error / np.pi,
        ], dtype=np.float32)
