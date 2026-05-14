"""
Teste live do PacSimEnv — requer PacSim a correr noutro terminal.

Mede:
  * tempo até à primeira perceção
  * latência de step (wall-clock e ROS-clock)
  * forma da observação e gama (deve estar contida em [-2, 2])
  * contagem de cones visíveis

Uso:
    # Terminal A
    source ~/pacsim_ws/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    ros2 launch pacsim example.launch.py

    # Terminal B
    source ~/pacsim_ws/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    python3 scripts/test_pacsim_live.py --steps 50
"""
import argparse
import os
import statistics
import sys
import time

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import gymnasium as gym

import env.pacsim_env  # noqa: F401  — regista 'PacSim-v0'


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--throttle', type=float, default=0.0,
                   help='throttle constante para a acção (0 = coast)')
    p.add_argument('--steering', type=float, default=0.0)
    p.add_argument('--startup-timeout', type=float, default=15.0)
    p.add_argument('--step-timeout', type=float, default=1.0)
    args = p.parse_args(argv)

    print(f'[live] criar PacSim-v0 (startup_timeout={args.startup_timeout}s)…')
    t0 = time.perf_counter()
    env = gym.make(
        'PacSim-v0',
        startup_timeout=args.startup_timeout,
        step_timeout=args.step_timeout,
    )
    t_ready = time.perf_counter() - t0
    print(f'[live] env pronto em {t_ready*1000:.1f} ms')

    obs, info = env.reset()
    assert obs.shape == (24,), f'shape inesperada: {obs.shape}'
    assert np.all(np.isfinite(obs)), 'obs contém NaN/Inf'
    assert np.all((obs >= -2.0) & (obs <= 2.0)), \
        f'obs fora de [-2, 2]: min={obs.min()} max={obs.max()}'
    print(f'[live] reset OK | obs.shape={obs.shape} | '
          f'n_cones_visible={info.get("n_cones_visible", "?")}')
    print(f'[live] ego[0:6] = {np.round(obs[:6], 4).tolist()}')

    action = np.array([args.steering, args.throttle], dtype=np.float32)
    latencies_ms = []
    stale_count = 0
    n_cones_min = 10**9
    n_cones_max = 0

    for i in range(args.steps):
        t = time.perf_counter()
        obs, _, _, _, info = env.step(action)
        dt_ms = (time.perf_counter() - t) * 1000.0
        latencies_ms.append(dt_ms)
        if info.get('stale_perception'):
            stale_count += 1
        n_cones = int(info.get('n_cones_visible', 0))
        n_cones_min = min(n_cones_min, n_cones)
        n_cones_max = max(n_cones_max, n_cones)

    env.close()

    if not latencies_ms:
        print('[live] sem dados — env.close() chamado sem steps')
        return 1

    median = statistics.median(latencies_ms)
    p95 = sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))]
    print()
    print(f'[live] {len(latencies_ms)} steps')
    print(f'[live] latência step (ms): '
          f'min={min(latencies_ms):.1f} med={median:.1f} '
          f'p95={p95:.1f} max={max(latencies_ms):.1f}')
    print(f'[live] step rate efectivo: ~{1000/median:.1f} Hz')
    print(f'[live] perception stale: {stale_count}/{len(latencies_ms)}')
    print(f'[live] cones visíveis por step: [{n_cones_min}, {n_cones_max}]')

    # Critério informal — sub-100ms é o alvo do meu próprio comentário anterior.
    if median > 100.0:
        print(f'[live] AVISO: latência mediana {median:.1f}ms > 100ms '
              f'(esperado para LiDAR a ~10Hz no PacSim).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
