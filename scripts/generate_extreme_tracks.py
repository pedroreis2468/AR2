"""
Gera 3 pistas extremas como held-out test set.

Estas pistas NÃO foram vistas durante treino nem usadas no test split inicial.
Servem como avaliação final de generalização.

  extreme_tight:  curvas apertadas em sucessão (raio mínimo no limite FS)
  extreme_fast:   pista grande, curvas suaves, top speed
  extreme_large:  pista muito comprida (~150m+), retas + curvas

Os ficheiros são guardados em tracks/extreme/*.yaml e podem ser carregados
pelo YAMLTrackLoader normal.

Uso:
  python scripts/generate_extreme_tracks.py
"""
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.track_generator import TrackGenerator, TrackParams


# Definição dos 3 perfis extremos
EXTREME_TRACKS = {
    'extreme_tight': dict(
        params=TrackParams(
            n_control_points=14,
            track_width=3.6,
            cone_spacing=3.5,
            min_radius=4.5,
            arena_size=50.0,
            noise_amplitude=0.30,
        ),
        seed=2026005,
        description="Curvas apertadas em sucessao; raio minimo FS (4.5m)",
    ),
    'extreme_fast': dict(
        params=TrackParams(
            n_control_points=8,
            track_width=4.5,
            cone_spacing=5.0,
            min_radius=12.0,
            arena_size=110.0,
            noise_amplitude=0.15,
        ),
        seed=2026002,
        description="Curvas largas e retas longas; ideal para high-speed",
    ),
    'extreme_large': dict(
        params=TrackParams(
            n_control_points=14,
            track_width=4.0,
            cone_spacing=4.0,
            min_radius=7.0,
            arena_size=140.0,
            noise_amplitude=0.35,
        ),
        seed=2026003,
        description="Pista longa (~180m); testa estabilidade em episodios extensos",
    ),
}


def track_to_yaml_dict(track_data, start_heading):
    """Converte output do TrackGenerator para formato YAML do PacSim."""
    blue_cones = track_data['blue_cones']
    yellow_cones = track_data['yellow_cones']
    orange_cones = track_data.get('orange_cones', np.zeros((0, 2)))

    left = [{'position': [float(c[0]), float(c[1]), 0.0], 'class': 'blue'}
            for c in blue_cones]
    right = [{'position': [float(c[0]), float(c[1]), 0.0], 'class': 'yellow'}
             for c in yellow_cones]
    time_keeping = [{'position': [float(c[0]), float(c[1]), 0.0],
                     'class': 'timekeeping'}
                    for c in orange_cones]

    start_pos = track_data['start_pos']

    return {
        'track': {
            'version': 1.0,
            'lanesFirstWithLastConnected': True,
            'start': {
                'position': [float(start_pos[0]), float(start_pos[1]), 0.0],
                'orientation': [0.0, 0.0, float(start_heading)],
            },
            'earthToTrack': {
                'position': [0.0, 0.0, 0.0],
                'orientation': [0.0, 0.0, 0.0],
            },
            'left': left,
            'right': right,
            'time_keeping': time_keeping,
            'unknown': [],
        }
    }


def main():
    out_dir = os.path.join('tracks', 'extreme')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print(" Geracao de pistas extremas (held-out test)")
    print("=" * 64)

    summary = []
    for name, cfg in EXTREME_TRACKS.items():
        print(f"\n[{name}] {cfg['description']}")
        p = cfg['params']
        print(f"  seed={cfg['seed']}, arena={p.arena_size}m, "
              f"width={p.track_width}m, min_r={p.min_radius}m")

        gen = TrackGenerator(params=p, seed=cfg['seed'])
        td = gen.generate()

        tangents = td['tangents']
        start_heading = float(np.arctan2(tangents[0, 1], tangents[0, 0]))

        yaml_dict = track_to_yaml_dict(td, start_heading)
        out_path = os.path.join(out_dir, f'{name}.yaml')
        with open(out_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(yaml_dict, f, sort_keys=False, default_flow_style=None)

        n_blue = len(td['blue_cones'])
        n_yellow = len(td['yellow_cones'])
        n_orange = len(td.get('orange_cones', []))
        print(f"  -> {n_blue} blue + {n_yellow} yellow + {n_orange} orange cones")
        print(f"  -> comprimento: {td['track_length']:.1f}m")
        print(f"  -> ficheiro: {out_path}")

        summary.append({
            'name': name,
            'length': td['track_length'],
            'cones_blue': n_blue,
            'cones_yellow': n_yellow,
        })

    print("\n" + "=" * 64)
    print(" Sumario")
    print("=" * 64)
    print(f"{'Pista':<18} {'Comprimento':>12} {'Cones (B/Y)':>14}")
    print('-' * 50)
    for s in summary:
        print(f"{s['name']:<18} {s['length']:>10.1f}m   "
              f"{s['cones_blue']:>4}/{s['cones_yellow']:<4}")
    print(f"\n[OK] {len(summary)} pistas guardadas em {out_dir}/")
    print("\nPara avaliar:")
    print("  python evaluate.py --mode sb3 \\")
    print("    --model runs/sac_seed2/best/best_model.zip \\")
    print("    --tracks-dir tracks/extreme --track extreme_tight")


if __name__ == '__main__':
    main()
