"""
debug_reward.py — Verifica componentes do reward sem treinar.

Uso:
    python debug_reward.py
    python debug_reward.py --tracks-dir ../pistas/tracks --steps 300
    python debug_reward.py --tracks-dir ../pistas/tracks --action left --steps 200
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.racing_env import FSRacingEnv


def main():
    parser = argparse.ArgumentParser(description="Debug reward components")
    parser.add_argument('--tracks-dir', type=str, default='tracks')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--action', type=str, default='forward',
                        choices=['forward', 'left', 'right', 'random'],
                        help='Ação fixa para testar')
    parser.add_argument('--track', type=str, default=None,
                        help='Nome da pista específica (ex: FSG19)')
    args = parser.parse_args()

    env = FSRacingEnv(
        tracks_dir=args.tracks_dir,
        render_mode=None,
        randomize_track=(args.track is None),
        domain_randomization=False,
        terminate_on_cone=False,
        max_episode_steps=args.steps + 1,
        use_orange_cones=True,
        track_name=args.track,
    )

    obs, info = env.reset()
    track_name = info.get('track_name', '?')

    # Calcular track half-width uma vez
    td = env.track_data
    effective_hw = td.get('track_width', 4.0) / 2.0

    print(f"\n{'='*80}")
    print(f"  Reward Debug — pista: {track_name}  |  hw={effective_hw:.1f}m  |  "
          f"len={td['track_length']:.0f}m")
    print(f"  Obs dim: {obs.shape[0]} | Ação: {args.action} | "
          f"Término fora-pista: {effective_hw*2.5:.1f}m lateral")
    print(f"{'='*80}")
    hdr = f"{'Step':>5} {'Steer':>6} {'Throt':>6} {'Rew':>7} " \
          f"{'ΣRew':>8} {'Speed':>7} {'ΔProg':>8} {'ΣProg':>8} " \
          f"{'HdgErr':>8} {'Latera':>8} {'r_lat':>7}"
    print(hdr)
    print('-' * 80)

    total_reward = 0.0
    cumulative_progress = 0.0
    window_progress = 0.0          # accumulated dp over the display window
    window_reward = 0.0            # accumulated reward over the display window

    def get_lateral_and_heading():
        """Compute lateral distance and heading error from current car state."""
        cl = td['centerline']
        car_pos = np.array([env.car.x, env.car.y])
        dists = np.sum((cl - car_pos)**2, axis=1)
        cl_idx = int(np.argmin(dists))
        # Lateral distance (signed: positive = left of centerline)
        to_car = car_pos - cl[cl_idx]
        lateral_signed = float(np.dot(to_car, td['normals'][cl_idx]))
        lateral_abs = abs(float(np.sqrt(dists[cl_idx])))
        # Heading error
        tangent = td['tangents'][cl_idx]
        track_hdg = np.arctan2(tangent[1], tangent[0])
        hdg_err = (env.car.theta - track_hdg + np.pi) % (2 * np.pi) - np.pi
        return lateral_abs, lateral_signed, float(hdg_err)

    prev_progress = 0.0
    PRINT_EVERY = 20

    for step in range(args.steps):
        # Build action
        if args.action == 'forward':
            action = np.array([0.0, 0.5])
        elif args.action == 'left':
            steer = min(0.6, step * 0.005)
            action = np.array([steer, 0.5])
        elif args.action == 'right':
            steer = -min(0.6, step * 0.005)
            action = np.array([steer, 0.5])
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        cur_progress = info.get('total_progress', 0.0)
        dp = cur_progress - prev_progress
        prev_progress = cur_progress
        cumulative_progress = cur_progress
        window_progress += dp
        window_reward += reward

        lateral_abs, lateral_signed, hdg_err = get_lateral_and_heading()

        # Compute r_lateral for display
        lat_ratio = lateral_abs / (effective_hw + 1e-6)
        if lat_ratio <= 1.0:
            r_lat = -lat_ratio * 0.08
        else:
            overshoot = lat_ratio - 1.0
            r_lat = -0.08 - overshoot ** 2 * 2.0

        # Print every PRINT_EVERY steps or on termination/truncation
        if step % PRINT_EVERY == 0 or terminated or truncated:
            flag = ""
            if terminated:
                flag = " ← TERM"
            elif truncated:
                flag = " ← TRUNC"
            print(f"{step:5d} {action[0]:+6.3f} {action[1]:+6.3f} "
                  f"{window_reward / max(1, PRINT_EVERY if step >= PRINT_EVERY else step+1):+7.3f} "
                  f"{total_reward:+8.2f} "
                  f"{info.get('speed_kmh', 0):6.1f}km "
                  f"{window_progress:+8.3f}m "
                  f"{cumulative_progress:+8.2f}m "
                  f"{np.degrees(hdg_err):+8.1f}° "
                  f"{lateral_signed:+8.3f}m "
                  f"{r_lat:+7.3f}{flag}")
            window_progress = 0.0
            window_reward = 0.0

        if terminated:
            print(f"\n  ⚠  Terminated at step {step}  "
                  f"(lateral={lateral_abs:.2f}m vs hw={effective_hw:.2f}m, "
                  f"limit={effective_hw*2.5:.2f}m)")
            break
        if truncated:
            print(f"\n  ✓  Truncated (max steps={args.steps})")
            break

    print('-' * 80)
    print(f"  Total reward:    {total_reward:+.2f}")
    print(f"  Total progress:  {info.get('total_progress', 0):+.1f} m  "
          f"(track length = {td['track_length']:.0f} m)")
    print(f"  Final speed:     {info.get('speed_kmh', 0):.1f} km/h")
    print(f"  Laps completed:  {info.get('laps_completed', 0)}")
    print()
    print(f"  NOTE: 'Rew' = avg reward per step in window | 'ΔProg' = window total progress")
    print()
    print("  REWARD WEIGHTS:")
    print("    r_progress  = Δprog × 2.0  (dominante: 1m ≈ +2 pts)")
    print("    r_alignment = (v/v_max) × cos(hdg_err) × 0.4")
    print("    r_smooth    = -max(0, |Δsteer| - 0.1) × 0.05  [dead-band]")
    print("    r_lateral   = -ratio×0.08 (dentro) / -0.08-(over²×2) (fora)")
    print("    r_time      = -0.005 por step")
    print("    cone hit    = -8.0 por cone (sem terminação por defeito)")
    print("    fora-pista  = -50.0 + terminated  (só >2.5× meia-largura)")
    print("    lap bonus   = +200.0 + terminated")
    print(f"{'='*80}\n")

    env.close()


if __name__ == '__main__':
    main()
