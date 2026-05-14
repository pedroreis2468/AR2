"""
Script de avaliação para agentes treinados.
Visualiza o agente a conduzir em pistas aleatórias.

Uso:
  python evaluate.py --model runs/custom_sac_.../best_model.pt --mode custom
  python evaluate.py --model runs/sb3_sac_.../final_model.zip --mode sb3
  python evaluate.py --random  # agente aleatório para teste
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.racing_env import FSRacingEnv


def _make_env(args):
    """
    Constrói o env de avaliação consoante args.env.

    PacSim exige ROS 2 sourced e PacSim a correr; falha com mensagem útil.
    """
    if args.env == 'pacsim':
        from env.pacsim_env import PacSimEnv
        return PacSimEnv(
            use_orange_cones=not args.legacy_obs,
            max_episode_steps=args.max_steps,
            startup_timeout=args.pacsim_startup_timeout,
            step_timeout=args.pacsim_step_timeout,
        )
    return FSRacingEnv(
        render_mode='human',
        randomize_track=(args.track is None),
        domain_randomization=False,
        max_episode_steps=args.max_steps,
        tracks_dir=args.tracks_dir,
        track_name=args.track,
        use_orange_cones=not args.legacy_obs,
        terminate_on_cone=False,
        doo_cone_limit=999,
        max_laps=args.max_laps,
    )


def _maybe_render(env, args):
    """Render só faz sentido na FSRacingEnv (PyGame). PacSim é via Foxglove."""
    if args.env == 'fs':
        return env.render()
    return True


def evaluate_custom(args):
    """Avalia agente SAC custom."""
    from agent.sac import SACAgent

    env = _make_env(args)

    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=args.device,
    )
    agent.load(args.model)

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep if args.seed else None)
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            _maybe_render(env, args)

        print(
            f"Ep {ep+1}/{args.n_episodes} | "
            f"Reward: {total_reward:.1f} | "
            f"Steps: {info.get('step', 0)} | "
            f"Progress: {info.get('total_progress', 0):.1f}m | "
            f"Laps: {info.get('laps_completed', 0)} | "
            f"Cones: {info.get('cones_hit', 0)}"
        )
        if terminated:
            reason = info.get('termination_reason', '??')
            track = info.get('track_name', '??')
            if 'volta completa' in reason:
                print(f"  ✅ Completo ({track}): {reason}")
            else:
                print(f"  ⛔ Terminado ({track}): {reason}")
        time.sleep(2)

    env.close()


def evaluate_sb3(args):
    """Avalia agente SB3 sem VecEnv para controlar rendering/reset manualmente."""
    try:
        from stable_baselines3 import SAC, PPO
    except ImportError:
        print("[ERRO] stable-baselines3 não instalado!")
        return

    env = _make_env(args)

    # Carregar modelo (norm_obs=False no treino, não precisa de VecNormalize)
    model_path = args.model
    if 'ppo' in args.model.lower():
        model = PPO.load(model_path, device=args.device)
    else:
        model = SAC.load(model_path, device=args.device)

    print(f"\n[INFO] Modelo: {args.model}")
    print(f"[INFO] Env: {args.env}")
    if args.env == 'fs':
        print(f"[INFO] Pista: {args.track or 'aleatória'}")
        print(f"[INFO] Max voltas: {args.max_laps}")
    print(f"[INFO] Episódios: {args.n_episodes}\n")
    print(f"{'Ep':>4} {'Reward':>8} {'Steps':>6} {'Progress':>10} {'Speed':>8} {'Laps':>5}")
    print('-' * 60)

    total_rewards = []

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep if args.seed else None)
        done = False
        total_reward = 0.0
        step = 0
        terminated = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            _maybe_render(env, args)

        total_rewards.append(total_reward)
        print(
            f"{ep+1:4d} "
            f"{total_reward:8.1f} "
            f"{step:6d} "
            f"{info.get('total_progress', 0):9.1f}m "
            f"{info.get('speed_kmh', 0):7.1f}km/h "
            f"{info.get('laps_completed', 0):5d} "
            f"Cones: {info.get('cones_hit', 0)}"
        )
        if terminated:
            reason = info.get('termination_reason', '??')
            track = info.get('track_name', '??')
            if 'volta completa' in reason:
                print(f"  ✅ Completo ({track}): {reason}")
            else:
                print(f"  ⛔ Terminado ({track}): {reason}")
        time.sleep(2)

    print('-' * 60)
    print(f"  Média: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    env.close()


def evaluate_random(args):
    """Corre o ambiente com ações aleatórias (para teste)."""
    env = _make_env(args)

    print(f"\n[INFO] A correr agente ALEATÓRIO no env '{args.env}'...")
    if args.env == 'fs':
        print("[INFO] Fechar a janela ou Ctrl+C para sair.\n")

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep if args.seed else None)
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = env.action_space.sample()
            action[1] = np.clip(action[1] + 0.3, -1, 1)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            if args.env == 'fs':
                result = env.render()
                if result is None:  # janela fechada
                    env.close()
                    return

        print(
            f"Ep {ep+1}/{args.n_episodes} | "
            f"Reward: {total_reward:.1f} | "
            f"Steps: {step} | "
            f"Progress: {info.get('total_progress', 0):.1f}m | "
            f"Cones: {info.get('cones_hit', 0)}"
        )
        if terminated:
            print("  ⛔ Episódio terminado — a mostrar local do acidente...")
        time.sleep(2)

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Avaliação de agentes FS Racing RL"
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Caminho para o modelo')
    parser.add_argument('--mode', type=str, default='custom',
                        choices=['custom', 'sb3'],
                        help='Tipo de modelo')
    parser.add_argument('--random', action='store_true',
                        help='Correr agente aleatório')
    parser.add_argument('--n-episodes', type=int, default=5,
                        help='Número de episódios')
    parser.add_argument('--max-steps', type=int, default=5000,
                        help='Máximo de steps por episódio')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed para reprodutibilidade')
    parser.add_argument('--tracks-dir', type=str, default='tracks',
                        help='Diretório com pistas YAML (default: ../pistas/tracks)')
    parser.add_argument('--track', type=str, default=None,
                        help='Nome de uma pista específica (e.g. FSG19)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Dispositivo (auto, cuda, cpu)')
    parser.add_argument('--legacy-obs', action='store_true',
                        help='Usar dimensão de observação 21 (para modelos antigos sem start/finish cones)')
    parser.add_argument('--no-terminate-on-cone', action='store_true',
                        help='Não terminar episódio imediatamente ao bater num cone')
    parser.add_argument('--max-laps', type=int, default=1,
                        help='Número máximo de voltas antes de terminar (default: 1)')
    parser.add_argument('--vecnormalize', type=str, default=None,
                        help='Caminho para vecnormalize.pkl (auto-detectado se na mesma pasta do modelo)')
    parser.add_argument('--env', type=str, default='fs',
                        choices=['fs', 'pacsim'],
                        help='Ambiente: fs (2D, default) ou pacsim (3D via ROS 2)')
    parser.add_argument('--pacsim-startup-timeout', type=float, default=15.0,
                        help='Timeout (s) à espera da primeira perceção do PacSim')
    parser.add_argument('--pacsim-step-timeout', type=float, default=1.0,
                        help='Timeout (s) por step à espera de perceção fresca')

    args = parser.parse_args()

    if args.random:
        evaluate_random(args)
    elif args.model is None:
        print("[ERRO] Especificar --model ou --random")
        parser.print_help()
    elif args.mode == 'custom':
        evaluate_custom(args)
    else:
        evaluate_sb3(args)


if __name__ == '__main__':
    main()
