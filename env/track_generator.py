"""
Gerador procedural de pistas Formula Student com cones.
Gera pistas fechadas com cones azuis (esquerda) e amarelos (direita),
seguindo as regras da FSG (Formula Student Germany).
"""
import numpy as np
from scipy.interpolate import CubicSpline
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class TrackParams:
    """Parâmetros de geração da pista."""
    n_control_points: int = 12     # pontos de controle da spline
    track_width: float = 4.0       # largura da pista (m) - FSG mínimo 3.5m
    cone_spacing: float = 4.0      # distância entre cones (m) - FSG ~3-5m
    min_radius: float = 4.5        # raio mínimo curva (m) - FSG spec
    arena_size: float = 80.0       # tamanho da arena (m)
    noise_amplitude: float = 0.3   # perturbação nos pontos de controle
    orange_cone_count: int = 4     # cones laranja na start/finish (2 por lado)


class TrackGenerator:
    """Gera pistas fechadas com cones azuis e amarelos."""

    def __init__(self, params: TrackParams = None, seed: Optional[int] = None):
        self.params = params or TrackParams()
        self.rng = np.random.RandomState(seed)

    def generate(self) -> dict:
        """
        Gera uma pista completa.
        Retorna dict com:
            - centerline: np.ndarray (N, 2) - linha central
            - blue_cones: np.ndarray (M, 2) - cones azuis (esquerda)
            - yellow_cones: np.ndarray (M, 2) - cones amarelos (direita)
            - orange_cones: np.ndarray (K, 2) - cones laranja (start/finish)
            - left_boundary: np.ndarray (N, 2)
            - right_boundary: np.ndarray (N, 2)
            - start_pos: np.ndarray (2,) - posição inicial
            - start_heading: float - heading inicial (rad)
            - track_length: float - comprimento total (m)
        """
        p = self.params

        # 1. Gerar pontos de controle em forma de loop
        control_pts = self._generate_control_points()

        # 2. Suavizar com spline cúbica fechada
        centerline = self._fit_closed_spline(control_pts, n_points=500)

        # 3. Calcular normais e boundaries
        tangents, normals = self._compute_tangents_normals(centerline)
        hw = p.track_width / 2.0
        left_boundary = centerline + normals * hw
        right_boundary = centerline - normals * hw

        # 4. Colocar cones ao longo dos boundaries
        blue_cones = self._place_cones_along_path(left_boundary, centerline)
        yellow_cones = self._place_cones_along_path(right_boundary, centerline)

        # 5. Cones laranja no start/finish
        orange_cones = self._place_start_finish_cones(
            centerline, left_boundary, right_boundary
        )

        # 6. Calcular posição e heading iniciais
        start_pos = centerline[0].copy()
        start_heading = np.arctan2(tangents[0, 1], tangents[0, 0])

        # 7. Comprimento total da pista
        diffs = np.diff(centerline, axis=0)
        track_length = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))

        return {
            'centerline': centerline,
            'blue_cones': blue_cones,
            'yellow_cones': yellow_cones,
            'orange_cones': orange_cones,
            'left_boundary': left_boundary,
            'right_boundary': right_boundary,
            'tangents': tangents,
            'normals': normals,
            'start_pos': start_pos,
            'start_heading': start_heading,
            'track_length': track_length,
        }

    def _generate_control_points(self) -> np.ndarray:
        """Gera pontos de controle distribuídos numa elipse perturbada."""
        p = self.params
        n = p.n_control_points

        # Base: elipse
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rx = p.arena_size * 0.35
        ry = p.arena_size * 0.25

        pts = np.zeros((n, 2))
        pts[:, 0] = rx * np.cos(angles)
        pts[:, 1] = ry * np.sin(angles)

        # Perturbar radialmente
        for i in range(n):
            r_noise = self.rng.uniform(-1, 1) * p.noise_amplitude * min(rx, ry)
            angle = angles[i]
            pts[i, 0] += r_noise * np.cos(angle)
            pts[i, 1] += r_noise * np.sin(angle)

        # Adicionar perturbação tangencial
        for i in range(n):
            tangent_noise = self.rng.uniform(-0.5, 0.5) * p.noise_amplitude * 5
            perp_angle = angles[i] + np.pi / 2
            pts[i, 0] += tangent_noise * np.cos(perp_angle)
            pts[i, 1] += tangent_noise * np.sin(perp_angle)

        return pts

    def _fit_closed_spline(self, control_pts: np.ndarray,
                           n_points: int = 500) -> np.ndarray:
        """Ajusta spline cúbica fechada aos pontos de controle."""
        # Fechar o loop adicionando o primeiro ponto ao final
        pts_closed = np.vstack([control_pts, control_pts[0:1]])
        n = len(pts_closed)
        t = np.arange(n, dtype=float)

        # Spline cúbica periódica
        cs_x = CubicSpline(t, pts_closed[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, pts_closed[:, 1], bc_type='periodic')

        t_fine = np.linspace(0, n - 1, n_points, endpoint=False)
        centerline = np.column_stack([cs_x(t_fine), cs_y(t_fine)])

        # Verificar e corrigir raio mínimo
        centerline = self._enforce_min_radius(centerline)

        return centerline

    def _enforce_min_radius(self, centerline: np.ndarray) -> np.ndarray:
        """Suaviza curvas demasiado apertadas (abaixo do raio mínimo)."""
        # Calcular curvatura
        dx = np.gradient(centerline[:, 0])
        dy = np.gradient(centerline[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        denom = (dx**2 + dy**2)**1.5
        denom = np.where(denom > 1e-10, denom, 1e-10)
        curvature = np.abs(dx * ddy - dy * ddx) / denom

        max_curvature = 1.0 / self.params.min_radius

        # Suavizar iterativamente onde curvatura é excessiva
        for _ in range(5):
            mask = curvature > max_curvature * 1.2
            if not mask.any():
                break
            # Aplicar filtro gaussiano local
            kernel_size = 7
            kernel = np.ones(kernel_size) / kernel_size
            for dim in range(2):
                padded = np.pad(centerline[:, dim], kernel_size, mode='wrap')
                smoothed = np.convolve(padded, kernel, mode='same')
                smoothed = smoothed[kernel_size:-kernel_size]
                centerline[mask, dim] = (
                    0.5 * centerline[mask, dim] + 0.5 * smoothed[mask]
                )
            # Recalcular curvatura
            dx = np.gradient(centerline[:, 0])
            dy = np.gradient(centerline[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            denom = (dx**2 + dy**2)**1.5
            denom = np.where(denom > 1e-10, denom, 1e-10)
            curvature = np.abs(dx * ddy - dy * ddx) / denom

        return centerline

    def _compute_tangents_normals(
        self, centerline: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula vetores tangente e normal em cada ponto da centerline."""
        n = len(centerline)
        tangents = np.zeros_like(centerline)
        normals = np.zeros_like(centerline)

        for i in range(n):
            # Diferenças centrais (circular)
            prev = centerline[(i - 1) % n]
            next_pt = centerline[(i + 1) % n]
            t = next_pt - prev
            mag = np.sqrt(t[0]**2 + t[1]**2)
            if mag > 1e-10:
                t /= mag
            tangents[i] = t
            # Normal: rotação 90° anti-horário (aponta para a esquerda)
            normals[i] = np.array([-t[1], t[0]])

        return tangents, normals

    def _place_cones_along_path(self, boundary: np.ndarray,
                                centerline: np.ndarray) -> np.ndarray:
        """Coloca cones ao longo de um boundary com espaçamento uniforme na pista fechada."""
        p = self.params
        
        # Calcular distância de cada segmento
        diffs = np.diff(boundary, axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Distância de fecho do loop
        close_dist = np.sqrt(np.sum((boundary[-1] - boundary[0])**2))
        
        # Distância total
        total_dist = np.sum(dists) + close_dist
        
        # Número de cones exatos para distribuir uniformemente (fechando o loop)
        num_cones = int(round(total_dist / p.cone_spacing))
        actual_spacing = total_dist / max(1, num_cones)
        
        # Distâncias acumuladas
        cum_dist = np.zeros(len(boundary) + 1)
        cum_dist[1:-1] = np.cumsum(dists)
        cum_dist[-1] = total_dist
        
        # Boundary com fecho para interpolação
        closed_boundary = np.vstack([boundary, boundary[0]])
        
        cones = []
        target_dists = np.arange(num_cones) * actual_spacing
        
        for tgt in target_dists:
            idx = int(np.searchsorted(cum_dist, tgt)) - 1
            idx = max(0, idx)
            
            segment_length = cum_dist[idx+1] - cum_dist[idx]
            if segment_length > 1e-6:
                t = (tgt - cum_dist[idx]) / segment_length
                cone_pos = closed_boundary[idx] * (1 - t) + closed_boundary[idx+1] * t
            else:
                cone_pos = closed_boundary[idx]
            cones.append(cone_pos)

        return np.array(cones)

    def _place_start_finish_cones(
        self, centerline: np.ndarray,
        left_boundary: np.ndarray,
        right_boundary: np.ndarray
    ) -> np.ndarray:
        """Coloca cones laranja na zona start/finish."""
        # 2 cones laranja de cada lado, perto do ponto 0
        cones = []
        for offset in [-1, 0]:
            idx = offset % len(centerline)
            cones.append(left_boundary[idx])
            cones.append(right_boundary[idx])
        return np.array(cones)

    def generate_batch(self, n_tracks: int) -> List[dict]:
        """Gera múltiplas pistas para treino diversificado."""
        tracks = []
        for i in range(n_tracks):
            self.rng = np.random.RandomState(self.rng.randint(0, 2**31))
            tracks.append(self.generate())
        return tracks


def compute_progress_along_centerline(
    pos: np.ndarray, centerline: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    Calcula o progresso de uma posição ao longo da centerline.
    Retorna: (idx_mais_proximo, distancia_lateral, heading_error_placeholder, distancia_acumulada)
    """
    diffs = centerline - pos
    dists_sq = np.sum(diffs**2, axis=1)
    idx = int(np.argmin(dists_sq))
    lateral_dist = np.sqrt(dists_sq[idx])

    # Distância acumulada até este índice
    if idx > 0:
        segments = np.diff(centerline[:idx + 1], axis=0)
        accumulated = float(np.sum(np.sqrt(np.sum(segments**2, axis=1))))
    else:
        accumulated = 0.0

    return idx, lateral_dist, 0.0, accumulated
