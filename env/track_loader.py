"""
Carregador de pistas Formula Student a partir de ficheiros YAML.
Lê os ficheiros de pista do formato pacsim/pistas e converte para o
formato interno usado pelo ambiente RL.
"""
import os
import glob
import yaml
import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Optional, Tuple


# Tracks que não são circuitos fechados (excluídas do treino)
_NON_CIRCUIT_TRACKS = {'skidpad', 'acceleration', 'gripMap'}


class YAMLTrackLoader:
    """Carrega e converte pistas YAML para o formato do ambiente RL."""

    def __init__(self, tracks_dir: str, exclude_non_circuits: bool = True):
        """
        Args:
            tracks_dir: Caminho para o diretório com ficheiros .yaml
            exclude_non_circuits: Se True, exclui skidpad/acceleration/gripMap
        """
        self.tracks_dir = tracks_dir
        self.exclude_non_circuits = exclude_non_circuits
        self._track_files: List[str] = []
        self._scan_tracks()

    def _scan_tracks(self):
        """Descobre todos os ficheiros .yaml no diretório."""
        pattern = os.path.join(self.tracks_dir, '*.yaml')
        all_files = sorted(glob.glob(pattern))

        if self.exclude_non_circuits:
            self._track_files = [
                f for f in all_files
                if os.path.splitext(os.path.basename(f))[0] not in _NON_CIRCUIT_TRACKS
            ]
        else:
            self._track_files = all_files

        if not self._track_files:
            raise FileNotFoundError(
                f"Nenhuma pista YAML encontrada em: {self.tracks_dir}"
            )

    @property
    def track_names(self) -> List[str]:
        """Lista de nomes de pistas disponíveis (sem extensão)."""
        return [os.path.splitext(os.path.basename(f))[0]
                for f in self._track_files]

    @property
    def n_tracks(self) -> int:
        return len(self._track_files)

    def load_track(self, name: Optional[str] = None,
                   index: Optional[int] = None) -> dict:
        """
        Carrega uma pista pelo nome ou índice.
        Retorna dict compatível com TrackGenerator.generate().

        Args:
            name: Nome da pista (e.g. 'FSG19')
            index: Índice na lista de pistas
        """
        if name is not None:
            filepath = os.path.join(self.tracks_dir, f'{name}.yaml')
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Pista não encontrada: {filepath}")
        elif index is not None:
            filepath = self._track_files[index % len(self._track_files)]
        else:
            raise ValueError("Especificar name ou index")

        return self._parse_yaml_track(filepath)

    def load_random(self, rng: np.random.RandomState = None) -> Tuple[dict, str]:
        """
        Carrega uma pista aleatória.
        Retorna (track_data, track_name).
        """
        if rng is None:
            rng = np.random.RandomState()
        idx = rng.randint(0, len(self._track_files))
        filepath = self._track_files[idx]
        name = os.path.splitext(os.path.basename(filepath))[0]
        return self._parse_yaml_track(filepath), name

    def _parse_yaml_track(self, filepath: str) -> dict:
        """Lê um ficheiro YAML e converte para o formato do ambiente."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        track = data['track']

        # --- Extrair cones por cor ---
        blue_cones = []
        yellow_cones = []
        orange_cones = []

        # Lado esquerdo (tipicamente blue)
        for cone in track.get('left', []):
            pos = cone['position'][:2]  # ignorar z
            cls = cone.get('class', '').lower()
            if cls == 'blue':
                blue_cones.append(pos)
            elif cls == 'yellow':
                yellow_cones.append(pos)
            elif cls in ('orange', 'small-orange', 'big-orange'):
                orange_cones.append(pos)
            elif cls == 'invisible':
                pass  # ignorar cones invisíveis
            else:
                # Classe desconhecida, atribuir ao lado esquerdo (blue)
                blue_cones.append(pos)

        # Lado direito (tipicamente yellow)
        for cone in track.get('right', []):
            pos = cone['position'][:2]
            cls = cone.get('class', '').lower()
            if cls == 'yellow':
                yellow_cones.append(pos)
            elif cls == 'blue':
                blue_cones.append(pos)
            elif cls in ('orange', 'small-orange', 'big-orange'):
                orange_cones.append(pos)
            elif cls == 'invisible':
                pass
            else:
                yellow_cones.append(pos)

        # Cones da secção 'unknown' (se existir)
        for cone in track.get('unknown', []):
            pos = cone['position'][:2]
            cls = cone.get('class', '').lower()
            if cls == 'blue':
                blue_cones.append(pos)
            elif cls == 'yellow':
                yellow_cones.append(pos)
            elif cls in ('orange', 'small-orange', 'big-orange'):
                orange_cones.append(pos)

        # Cones de timekeeping como orange
        for cone in track.get('time_keeping', []):
            pos = cone['position'][:2]
            orange_cones.append(pos)

        blue_cones = np.array(blue_cones, dtype=np.float64)
        yellow_cones = np.array(yellow_cones, dtype=np.float64)
        orange_cones = (np.array(orange_cones, dtype=np.float64)
                        if orange_cones else np.zeros((0, 2)))

        # Se não há cones laranja, criar a partir dos primeiros cones
        if len(orange_cones) == 0 and len(blue_cones) > 0 and len(yellow_cones) > 0:
            orange_cones = np.array([
                blue_cones[0], yellow_cones[0],
                blue_cones[-1], yellow_cones[-1],
            ])

        # --- Extrair posição/orientação de partida ---
        start = track.get('start', {})
        start_position = np.array(
            start.get('position', [0.0, 0.0, 0.0])[:2], dtype=np.float64
        )
        start_orientation = start.get('orientation', [0.0, 0.0, 0.0])
        # A orientação no YAML é em eixos (roll, pitch, yaw) = z-up
        start_heading = float(start_orientation[2])  # yaw

        # --- Construir boundaries ordenados (left/right do YAML) ---
        left_boundary = self._extract_ordered_boundary(track.get('left', []))
        right_boundary = self._extract_ordered_boundary(track.get('right', []))

        # --- Calcular centerline ---
        connected = track.get('lanesFirstWithLastConnected', True)
        centerline = self._compute_centerline(
            left_boundary, right_boundary, closed=connected
        )

        # --- Calcular tangentes e normais ---
        tangents, normals = self._compute_tangents_normals(centerline)

        # --- Recalcular left/right boundaries a partir da centerline ---
        # Usar os boundaries reais (cones) em vez de gerar a partir da centerline
        # para manter fidelidade ao layout real
        left_boundary_smooth = self._resample_boundary(
            left_boundary, len(centerline)
        )
        right_boundary_smooth = self._resample_boundary(
            right_boundary, len(centerline)
        )

        # --- Calcular largura média da pista ---
        track_width = self._estimate_track_width(
            centerline, left_boundary, right_boundary
        )

        # --- Heading inicial ---
        # Se start_heading == 0, calcular a partir da tangente da centerline
        if abs(start_heading) < 1e-6 and len(tangents) > 0:
            start_heading = float(np.arctan2(tangents[0, 1], tangents[0, 0]))

        # --- Comprimento da pista ---
        diffs = np.diff(centerline, axis=0)
        track_length = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))

        return {
            'centerline': centerline,
            'blue_cones': blue_cones,
            'yellow_cones': yellow_cones,
            'orange_cones': orange_cones,
            'left_boundary': left_boundary_smooth,
            'right_boundary': right_boundary_smooth,
            'tangents': tangents,
            'normals': normals,
            'start_pos': start_position.copy(),
            'start_heading': start_heading,
            'track_length': track_length,
            'track_width': track_width,
        }

    @staticmethod
    def _extract_ordered_boundary(cones_list: list) -> np.ndarray:
        """Extrai posições 2D ordenadas de uma lista de cones YAML."""
        positions = []
        for cone in cones_list:
            cls = cone.get('class', '').lower()
            if cls == 'invisible':
                continue
            positions.append(cone['position'][:2])
        return np.array(positions, dtype=np.float64) if positions else np.zeros((0, 2))

    @staticmethod
    def _compute_centerline(
        left: np.ndarray, right: np.ndarray,
        closed: bool = True, n_points: int = 500
    ) -> np.ndarray:
        """
        Calcula a centerline suavizada a partir dos boundaries esquerdo e direito.
        Interpola ambos os lados para o mesmo número de pontos e faz a média.
        """
        if len(left) < 2 or len(right) < 2:
            raise ValueError("Pista precisa de pelo menos 2 cones de cada lado")

        # Parametrizar cada boundary pelo comprimento de arco acumulado
        left_interp = YAMLTrackLoader._parametrize_by_arclength(left)
        right_interp = YAMLTrackLoader._parametrize_by_arclength(right)

        # Interpolar ambos para n_raw pontos uniformes
        n_raw = max(len(left), len(right)) * 5
        t_left = np.linspace(0, 1, n_raw)
        t_right = np.linspace(0, 1, n_raw)

        left_pts = left_interp(t_left)
        right_pts = right_interp(t_right)

        # Centerline = média dos dois lados
        raw_center = (left_pts + right_pts) / 2.0

        # Suavizar com spline
        if closed and len(raw_center) > 3:
            centerline = YAMLTrackLoader._fit_closed_spline(
                raw_center, n_points
            )
        else:
            # Pista aberta - spline simples
            t = np.linspace(0, 1, len(raw_center))
            t_fine = np.linspace(0, 1, n_points)
            cs_x = CubicSpline(t, raw_center[:, 0])
            cs_y = CubicSpline(t, raw_center[:, 1])
            centerline = np.column_stack([cs_x(t_fine), cs_y(t_fine)])

        return centerline

    @staticmethod
    def _parametrize_by_arclength(pts: np.ndarray):
        """
        Cria interpolação parametrizada pelo comprimento de arco normalizado [0,1].
        Retorna callable(t) -> (N, 2).
        """
        diffs = np.diff(pts, axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        cumlen = np.concatenate([[0], np.cumsum(dists)])
        total = cumlen[-1]
        if total < 1e-10:
            total = 1.0
        t_param = cumlen / total

        # Garantir monotonia estrita
        for i in range(1, len(t_param)):
            if t_param[i] <= t_param[i-1]:
                t_param[i] = t_param[i-1] + 1e-10

        cs_x = CubicSpline(t_param, pts[:, 0])
        cs_y = CubicSpline(t_param, pts[:, 1])

        def interp(t):
            return np.column_stack([cs_x(t), cs_y(t)])

        return interp

    @staticmethod
    def _fit_closed_spline(pts: np.ndarray,
                           n_points: int = 500) -> np.ndarray:
        """Ajusta spline cúbica fechada aos pontos."""
        # Fechar o loop
        pts_closed = np.vstack([pts, pts[0:1]])
        n = len(pts_closed)
        t = np.arange(n, dtype=float)

        cs_x = CubicSpline(t, pts_closed[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, pts_closed[:, 1], bc_type='periodic')

        t_fine = np.linspace(0, n - 1, n_points, endpoint=False)
        return np.column_stack([cs_x(t_fine), cs_y(t_fine)])

    @staticmethod
    def _compute_tangents_normals(
        centerline: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula vetores tangente e normal em cada ponto da centerline."""
        n = len(centerline)
        tangents = np.zeros_like(centerline)
        normals = np.zeros_like(centerline)

        for i in range(n):
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

    @staticmethod
    def _resample_boundary(boundary: np.ndarray,
                           n_points: int) -> np.ndarray:
        """Reamostra um boundary para n_points pontos uniformes."""
        if len(boundary) < 2:
            return np.zeros((n_points, 2))

        interp = YAMLTrackLoader._parametrize_by_arclength(boundary)
        t = np.linspace(0, 1, n_points)
        return interp(t)

    @staticmethod
    def _estimate_track_width(
        centerline: np.ndarray,
        left_boundary: np.ndarray,
        right_boundary: np.ndarray
    ) -> float:
        """Estima a largura média da pista."""
        # Amostrar distâncias em vários pontos da centerline
        n_samples = min(50, len(centerline))
        indices = np.linspace(0, len(centerline) - 1, n_samples, dtype=int)

        widths = []
        for idx in indices:
            pt = centerline[idx]
            # Distância ao boundary esquerdo mais próximo
            if len(left_boundary) > 0:
                d_left = np.min(np.sqrt(np.sum(
                    (left_boundary - pt)**2, axis=1
                )))
            else:
                d_left = 2.0
            # Distância ao boundary direito mais próximo
            if len(right_boundary) > 0:
                d_right = np.min(np.sqrt(np.sum(
                    (right_boundary - pt)**2, axis=1
                )))
            else:
                d_right = 2.0
            widths.append(d_left + d_right)

        return float(np.median(widths))
