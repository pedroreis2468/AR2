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
from env.car_model import VehicleParams
from env.track_generator import TrackParams


def evaluate_custom(args):
    """Avalia agente SAC custom."""
    from agent.sac import SACAgent

    env = FSRacingEnv(
        render_mode='human',
        randomize_track=True,
        domain_randomization=False,
        max_episode_steps=args.max_steps,
    )

    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
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
            env.render()

        print(
            f"Ep {ep+1}/{args.n_episodes} | "
            f"Reward: {total_reward:.1f} | "
            f"Steps: {info['step']} | "
            f"Progress: {info['total_progress']:.1f}m | "
            f"Laps: {info['laps_completed']}"
        )

    env.close()


def evaluate_sb3(args):
    """Avalia agente SB3."""
    try:
        from stable_baselines3 import SAC, PPO
    except ImportError:
        print("[ERRO] stable-baselines3 não instalado!")
        return

    env = FSRacingEnv(
        render_mode='human',
        randomize_track=True,
        domain_randomization=False,
        max_episode_steps=args.max_steps,
    )

    # Detetar tipo de modelo
    if 'ppo' in args.model.lower():
        model = PPO.load(args.model)
    else:
        model = SAC.load(args.model)

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep if args.seed else None)
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()

        print(
            f"Ep {ep+1}/{args.n_episodes} | "
            f"Reward: {total_reward:.1f} | "
            f"Steps: {info['step']} | "
            f"Progress: {info['total_progress']:.1f}m | "
            f"Laps: {info['laps_completed']}"
        )

    env.close()


def evaluate_random(args):
    """Corre o ambiente com ações aleatórias (para teste)."""
    env = FSRacingEnv(
        render_mode='human',
        randomize_track=True,
        domain_randomization=False,
        max_episode_steps=args.max_steps,
    )

    print("\n[INFO] A correr agente ALEATÓRIO para testar o ambiente...")
    print("[INFO] Fechar a janela ou Ctrl+C para sair.\n")

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep if args.seed else None)
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = env.action_space.sample()
            # Bias ligeiro para a frente
            action[1] = np.clip(action[1] + 0.3, -1, 1)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            result = env.render()
            if result is None:  # janela fechada
                env.close()
                return

        print(
            f"Ep {ep+1}/{args.n_episodes} | "
            f"Reward: {total_reward:.1f} | "
            f"Steps: {step} | "
            f"Progress: {info['total_progress']:.1f}m"
        )

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
