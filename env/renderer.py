"""
Renderizador PyGame para o ambiente de corrida Formula Student.
Vista top-down com cones coloridos, carro, sensores e HUD.
"""
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class FSRenderer:
    """Renderiza o ambiente FS Racing com PyGame."""

    # Cores
    BG_COLOR = (30, 30, 35)
    TRACK_COLOR = (60, 60, 65)
    CENTERLINE_COLOR = (80, 80, 90)
    BLUE_CONE = (30, 120, 255)
    YELLOW_CONE = (255, 220, 40)
    ORANGE_CONE = (255, 140, 30)
    CAR_COLOR = (220, 50, 50)
    CAR_FRONT = (255, 255, 100)
    TRAJECTORY_COLOR = (100, 200, 100, 128)
    SENSOR_BLUE = (30, 120, 255, 100)
    SENSOR_YELLOW = (255, 220, 40, 100)
    SENSOR_ORANGE = (255, 140, 30, 100)
    HUD_BG = (20, 20, 25, 200)
    HUD_TEXT = (220, 220, 220)
    SPEED_BAR = (50, 200, 100)
    REWARD_POS = (50, 200, 100)
    REWARD_NEG = (200, 50, 50)

    def __init__(self, env, width: int = 1200, height: int = 800):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame não está instalado: pip install pygame")

        self.env = env
        self.width = width
        self.height = height

        pygame.init()
        if env.render_mode == 'human':
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Formula Student RL Racing")
        else:
            self.screen = pygame.Surface((width, height))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('monospace', 14)
        self.font_large = pygame.font.SysFont('monospace', 18, bold=True)
        self.font_title = pygame.font.SysFont('monospace', 22, bold=True)

        # Camera
        self._setup_camera()

    def _setup_camera(self):
        """Configura a câmara para enquadrar a pista toda."""
        td = self.env.track_data
        if td is None:
            return

        cl = td['centerline']
        all_cones = np.vstack([
            td['blue_cones'], td['yellow_cones'], td['orange_cones']
        ])
        all_pts = np.vstack([cl, all_cones])

        x_min, y_min = all_pts.min(axis=0) - 5
        x_max, y_max = all_pts.max(axis=0) + 5

        world_w = x_max - x_min
        world_h = y_max - y_min

        # Escala para caber no ecrã (com margem para HUD)
        render_w = self.width - 250  # margem para HUD
        render_h = self.height - 40
        self.scale = min(render_w / world_w, render_h / world_h)
        self.offset_x = -x_min * self.scale + 20
        self.offset_y = -y_min * self.scale + 20

    def world_to_screen(self, x: float, y: float) -> tuple:
        """Converte coordenadas do mundo para coordenadas do ecrã."""
        sx = int(x * self.scale + self.offset_x)
        sy = int(self.height - (y * self.scale + self.offset_y))
        return (sx, sy)

    def render(self) -> np.ndarray:
        """Renderiza um frame."""
        # Handle pygame events
        if self.env.render_mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        self._setup_camera()
        self.screen.fill(self.BG_COLOR)

        td = self.env.track_data
        if td is None:
            return None

        # 1. Desenhar pista (área entre boundaries)
        self._draw_track_surface(td)

        # 2. Centerline (tracejada)
        self._draw_centerline(td['centerline'])

        # 3. Trajectória do carro
        self._draw_trajectory()

        # 4. Cones
        self._draw_cones(td['blue_cones'], self.BLUE_CONE, 4)
        self._draw_cones(td['yellow_cones'], self.YELLOW_CONE, 4)
        self._draw_cones(td['orange_cones'], self.ORANGE_CONE, 6)

        # 5. Linhas de sensor (cones visíveis)
        self._draw_sensor_lines()

        # 6. Carro
        self._draw_car()

        # 7. HUD
        self._draw_hud()

        if self.env.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(50)

        return np.transpose(
            pygame.surfarray.array3d(self.screen), (1, 0, 2)
        )

    def _draw_track_surface(self, td):
        """Desenha a superfície da pista como polígono."""
        left = td['left_boundary']
        right = td['right_boundary']

        # Criar polígono: esquerda + direita invertida
        n = min(len(left), len(right))
        step = max(1, n // 200)  # reduzir pontos para performance

        pts_left = [self.world_to_screen(left[i, 0], left[i, 1])
                     for i in range(0, n, step)]
        pts_right = [self.world_to_screen(right[i, 0], right[i, 1])
                      for i in range(n - 1, -1, -step)]

        if len(pts_left) > 2 and len(pts_right) > 2:
            polygon = pts_left + pts_right
            pygame.draw.polygon(self.screen, self.TRACK_COLOR, polygon)
            
            # Preencher o gap na linha de partida/chegada
            p1 = self.world_to_screen(left[-1, 0], left[-1, 1])
            p2 = self.world_to_screen(right[-1, 0], right[-1, 1])
            p3 = self.world_to_screen(right[0, 0], right[0, 1])
            p4 = self.world_to_screen(left[0, 0], left[0, 1])
            pygame.draw.polygon(self.screen, self.TRACK_COLOR, [p1, p2, p3, p4])

    def _draw_centerline(self, centerline):
        """Desenha a centerline tracejada."""
        n = len(centerline)
        step = max(1, n // 150)
        for i in range(0, n - step, step * 2):
            p1 = self.world_to_screen(centerline[i, 0], centerline[i, 1])
            j = min(i + step, n - 1)
            p2 = self.world_to_screen(centerline[j, 0], centerline[j, 1])
            pygame.draw.line(self.screen, self.CENTERLINE_COLOR, p1, p2, 1)

    def _draw_trajectory(self):
        """Desenha a trajectória percorrida pelo carro."""
        traj = self.env._trajectory
        if len(traj) < 2:
            return

        # Desenhar últimos 500 pontos
        start = max(0, len(traj) - 500)
        step = max(1, (len(traj) - start) // 200)
        pts = [self.world_to_screen(x, y) for x, y in traj[start::step]]
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, self.TRAJECTORY_COLOR[:3], False, pts, 2)

    def _draw_cones(self, cones, color, radius):
        """Desenha cones como círculos."""
        for cone in cones:
            pos = self.world_to_screen(cone[0], cone[1])
            pygame.draw.circle(self.screen, color, pos, radius)
            # Borda mais escura
            darker = tuple(max(0, c - 40) for c in color)
            pygame.draw.circle(self.screen, darker, pos, radius, 1)

    def _draw_sensor_lines(self):
        """Desenha linhas dos sensores para os cones visíveis."""
        env = self.env
        if env.track_data is None:
            return

        car_pos = self.world_to_screen(env.car.x, env.car.y)

        blue_obs, yellow_obs, orange_obs, _ = env.cone_sensor.get_observations(
            env.car.x, env.car.y, env.car.theta,
            env.track_data['blue_cones'], env.track_data['yellow_cones'],
            env.track_data['orange_cones'],
            add_noise=False
        )

        # Converter de referencial do carro para mundo
        cos_t = np.cos(env.car.theta)
        sin_t = np.sin(env.car.theta)

        for obs, color in [(blue_obs, self.SENSOR_BLUE), (yellow_obs, self.SENSOR_YELLOW), (orange_obs, self.SENSOR_ORANGE)]:
            for i in range(len(obs)):
                if np.any(obs[i] != 0):
                    # Transformar de car frame para world frame
                    wx = env.car.x + obs[i, 0] * cos_t - obs[i, 1] * sin_t
                    wy = env.car.y + obs[i, 0] * sin_t + obs[i, 1] * cos_t
                    cone_screen = self.world_to_screen(wx, wy)
                    pygame.draw.line(self.screen, color[:3], car_pos, cone_screen, 1)

    def _transform_local_pts(self, local_pts, car_x, car_y, car_theta):
        """Transforma pontos locais (x-frente, y-esquerda) para coordenadas de ecrã."""
        cos_t = np.cos(car_theta)
        sin_t = np.sin(car_theta)
        screen_pts = []
        for lx, ly in local_pts:
            wx = car_x + lx * cos_t - ly * sin_t
            wy = car_y + lx * sin_t + ly * cos_t
            screen_pts.append(self.world_to_screen(wx, wy))
        return screen_pts

    def _draw_car(self):
        """Desenha o carro como vista top-down de um Formula Student."""
        env = self.env
        cx, cy, theta = env.car.x, env.car.y, env.car.theta
        p = env.vehicle_params

        # Dimensões reais do carro (em metros)
        L = p.length       # 3.0 m comprimento total
        W = p.width         # 1.4 m largura total
        hw = W / 2          # 0.7 m meia-largura

        # --- Corpo principal (monocoque afilado) ---
        # Pontos no referencial local: (x=frente, y=esquerda)
        body = [
            ( L * 0.45,  hw * 0.30),   # ponta do nariz (estreita)
            ( L * 0.35,  hw * 0.55),   # alargamento nariz
            ( L * 0.20,  hw * 0.70),   # início cockpit
            ( L * 0.05,  hw * 0.80),   # cockpit max width
            (-L * 0.10,  hw * 0.85),   # atrás cockpit
            (-L * 0.25,  hw * 0.80),   # sidepod zone
            (-L * 0.40,  hw * 0.70),   # traseira
            (-L * 0.48,  hw * 0.55),   # traseira estreita
            (-L * 0.48, -hw * 0.55),   # simétrico
            (-L * 0.40, -hw * 0.70),
            (-L * 0.25, -hw * 0.80),
            (-L * 0.10, -hw * 0.85),
            ( L * 0.05, -hw * 0.80),
            ( L * 0.20, -hw * 0.70),
            ( L * 0.35, -hw * 0.55),
            ( L * 0.45, -hw * 0.30),
        ]
        body_pts = self._transform_local_pts(body, cx, cy, theta)
        pygame.draw.polygon(self.screen, self.CAR_COLOR, body_pts)
        pygame.draw.polygon(self.screen, (180, 35, 35), body_pts, 2)

        # --- Cockpit (abertura do piloto) ---
        cockpit = [
            ( L * 0.12,  hw * 0.30),
            ( L * 0.05,  hw * 0.40),
            (-L * 0.10,  hw * 0.40),
            (-L * 0.15,  hw * 0.25),
            (-L * 0.15, -hw * 0.25),
            (-L * 0.10, -hw * 0.40),
            ( L * 0.05, -hw * 0.40),
            ( L * 0.12, -hw * 0.30),
        ]
        cockpit_pts = self._transform_local_pts(cockpit, cx, cy, theta)
        pygame.draw.polygon(self.screen, (40, 40, 45), cockpit_pts)

        # --- Asa dianteira ---
        fw_y = hw * 1.10  # mais larga que o corpo
        front_wing = [
            ( L * 0.48,  fw_y),
            ( L * 0.44,  fw_y),
            ( L * 0.44, -fw_y),
            ( L * 0.48, -fw_y),
        ]
        fw_pts = self._transform_local_pts(front_wing, cx, cy, theta)
        pygame.draw.polygon(self.screen, (200, 200, 210), fw_pts)
        # Endplates
        for sign in [1, -1]:
            ep = [
                ( L * 0.49,  sign * fw_y),
                ( L * 0.43,  sign * fw_y),
                ( L * 0.43,  sign * (fw_y - 0.05)),
                ( L * 0.49,  sign * (fw_y - 0.05)),
            ]
            ep_pts = self._transform_local_pts(ep, cx, cy, theta)
            pygame.draw.polygon(self.screen, (180, 180, 190), ep_pts)

        # --- Asa traseira ---
        rw_y = hw * 0.85
        rear_wing = [
            (-L * 0.46,  rw_y),
            (-L * 0.50,  rw_y),
            (-L * 0.50, -rw_y),
            (-L * 0.46, -rw_y),
        ]
        rw_pts = self._transform_local_pts(rear_wing, cx, cy, theta)
        pygame.draw.polygon(self.screen, (200, 200, 210), rw_pts)
        # Endplates
        for sign in [1, -1]:
            ep = [
                (-L * 0.45,  sign * rw_y),
                (-L * 0.51,  sign * rw_y),
                (-L * 0.51,  sign * (rw_y + 0.08)),
                (-L * 0.45,  sign * (rw_y + 0.08)),
            ]
            ep_pts = self._transform_local_pts(ep, cx, cy, theta)
            pygame.draw.polygon(self.screen, (180, 180, 190), ep_pts)

        # --- Rodas (4 rodas) ---
        wheel_w = 0.12   # largura da roda (m)
        wheel_l = 0.28   # comprimento da roda (m)
        wheel_color = (30, 30, 30)
        wheel_rim = (80, 80, 80)

        # Front wheels (com steering)
        steer = env.car.steering
        cos_s = np.cos(steer)
        sin_s = np.sin(steer)
        for sign in [1, -1]:
            wy = sign * hw * 0.95
            wx = L * 0.30  # posição longitudinal rodas dianteiras
            # Roda retangular local (rotacionada pelo steering)
            wheel_local = [
                ( wheel_l / 2,  wheel_w / 2),
                (-wheel_l / 2,  wheel_w / 2),
                (-wheel_l / 2, -wheel_w / 2),
                ( wheel_l / 2, -wheel_w / 2),
            ]
            # Rodar pelo ângulo de steering e posicionar
            rotated = []
            for wlx, wly in wheel_local:
                rx = wlx * cos_s - wly * sin_s + wx
                ry = wlx * sin_s + wly * cos_s + wy
                rotated.append((rx, ry))
            w_pts = self._transform_local_pts(rotated, cx, cy, theta)
            pygame.draw.polygon(self.screen, wheel_color, w_pts)
            pygame.draw.polygon(self.screen, wheel_rim, w_pts, 1)

        # Rear wheels (fixas)
        for sign in [1, -1]:
            wy = sign * hw * 0.90
            wx = -L * 0.35
            wheel_local = [
                (wx + wheel_l / 2, wy + wheel_w / 2),
                (wx - wheel_l / 2, wy + wheel_w / 2),
                (wx - wheel_l / 2, wy - wheel_w / 2),
                (wx + wheel_l / 2, wy - wheel_w / 2),
            ]
            w_pts = self._transform_local_pts(wheel_local, cx, cy, theta)
            pygame.draw.polygon(self.screen, wheel_color, w_pts)
            pygame.draw.polygon(self.screen, wheel_rim, w_pts, 1)

        # --- Número / marcação no nariz ---
        nose_marker = [
            ( L * 0.38,  hw * 0.15),
            ( L * 0.30,  hw * 0.15),
            ( L * 0.30, -hw * 0.15),
            ( L * 0.38, -hw * 0.15),
        ]
        nm_pts = self._transform_local_pts(nose_marker, cx, cy, theta)
        pygame.draw.polygon(self.screen, (255, 220, 50), nm_pts)

    def _draw_hud(self):
        """Desenha o HUD com informações do episódio."""
        env = self.env
        info = env._get_info()

        hud_x = self.width - 220
        hud_y = 10
        hud_w = 210
        hud_h = 380

        # Fundo do HUD
        hud_surface = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surface.fill(self.HUD_BG)
        self.screen.blit(hud_surface, (hud_x, hud_y))

        # Título
        title = self.font_title.render("FS Racing RL", True, (255, 200, 50))
        self.screen.blit(title, (hud_x + 10, hud_y + 10))

        # Nome da pista
        track_name = info.get('track_name', '')
        if track_name:
            tn_txt = self.font_large.render(track_name, True, (120, 200, 255))
            self.screen.blit(tn_txt, (hud_x + 10, hud_y + 36))

        y = hud_y + 62
        line_h = 22

        lines = [
            f"Step: {info['step']}",
            f"Speed: {info['speed_kmh']:.1f} km/h",
            f"Progress: {info['total_progress']:.1f} m",
            f"Laps: {info['laps_completed']}",
            f"Reward: {info['episode_reward']:.1f}",
            f"Steering: {info['steering']:.2f} rad",
            f"Heading: {np.degrees(info['theta']):.1f}°",
            f"Pos: ({info['x']:.1f}, {info['y']:.1f})",
        ]

        for line in lines:
            txt = self.font.render(line, True, self.HUD_TEXT)
            self.screen.blit(txt, (hud_x + 10, y))
            y += line_h

        # Barra de velocidade
        y += 10
        speed_pct = info['speed'] / env.vehicle_params.max_speed
        bar_w = hud_w - 20
        bar_h = 12

        pygame.draw.rect(self.screen, (50, 50, 55),
                         (hud_x + 10, y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.SPEED_BAR,
                         (hud_x + 10, y, int(bar_w * speed_pct), bar_h))
        txt = self.font.render("Speed", True, self.HUD_TEXT)
        self.screen.blit(txt, (hud_x + 10, y - 16))

        # Barra de reward
        y += 30
        reward_norm = np.clip(info['episode_reward'] / 500, -1, 1)
        color = self.REWARD_POS if reward_norm >= 0 else self.REWARD_NEG
        rw = int(abs(reward_norm) * bar_w / 2)
        center_x = hud_x + 10 + bar_w // 2

        pygame.draw.rect(self.screen, (50, 50, 55),
                         (hud_x + 10, y, bar_w, bar_h))
        if reward_norm >= 0:
            pygame.draw.rect(self.screen, color,
                             (center_x, y, rw, bar_h))
        else:
            pygame.draw.rect(self.screen, color,
                             (center_x - rw, y, rw, bar_h))

        txt = self.font.render("Reward", True, self.HUD_TEXT)
        self.screen.blit(txt, (hud_x + 10, y - 16))

    def close(self):
        """Fecha o renderer."""
        if PYGAME_AVAILABLE:
            pygame.quit()
