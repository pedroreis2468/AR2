"""
Script de treino para o agente FS Racing RL.
Suporta dois modos:
  1. SAC custom (agent/sac.py) - para fins educativos
  2. Stable-Baselines3 SAC/PPO - para resultados de produção

Uso:
  python train.py --mode custom --total-steps 500000
  python train.py --mode sb3 --algo sac --total-steps 1000000
  python train.py --mode sb3 --algo ppo --total-steps 1000000
"""
import argparse
import os
import time
import numpy as np
from datetime import datetime

# Adicionar path do projeto
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.racing_env import FSRacingEnv
from env.car_model import VehicleParams
from env.track_generator import TrackParams


def train_custom_sac(args):
    """Treino com implementação custom de SAC."""
    from agent.sac import SACAgent

    print("=" * 60)
    print("  Formula Student RL - Treino SAC Custom")
    print("=" * 60)

    # Criar ambiente
    env = FSRacingEnv(
        render_mode='human' if args.render else None,
        randomize_track=True,
        domain_randomization=True,
        max_episode_steps=args.max_ep_steps,
    )

    # Criar agente
    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dims=(256, 256),
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=1_000_000,
        batch_size=256,
        learning_starts=5000,
    )

    # Carregar checkpoint se existir
    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)

    # Diretório de logs
    log_dir = f"runs/custom_sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)

    # Loop de treino
    episode = 0
    best_reward = -float('inf')
    episode_rewards = []
    episode_lengths = []
    episode_cones_hit = []
    episode_time_penalties = []
    episode_progress = []
    start_time = time.time()

    obs, info = env.reset()

    while agent.total_steps < args.total_steps:
        # Selecionar ação
        if agent.total_steps < agent.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        # Executar ação
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Armazenar transição
        agent.store_transition(obs, action, reward, next_obs, terminated)

        # Atualizar agente
        if agent.total_steps >= agent.learning_starts:
            metrics = agent.update()
        else:
            metrics = {}

        obs = next_obs

        if done:
            episode += 1
            ep_reward = info['episode_reward']
            ep_length = info['step']
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            episode_cones_hit.append(info.get('cones_hit', 0))
            episode_time_penalties.append(info.get('time_penalty', 0.0))
            episode_progress.append(info.get('total_progress', 0.0))

            # Log
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_length = np.mean(episode_lengths[-50:])
                elapsed = time.time() - start_time
                fps = agent.total_steps / elapsed

                print(
                    f"Ep {episode:5d} | "
                    f"Steps {agent.total_steps:8d} | "
                    f"Reward {ep_reward:8.1f} | "
                    f"Avg50 {avg_reward:8.1f} | "
                    f"Len {ep_length:5d} | "
                    f"Speed {info['speed_kmh']:.1f}km/h | "
                    f"Cones {info.get('cones_hit', 0):2d} | "
                    f"α {metrics.get('alpha', 0):.3f} | "
                    f"FPS {fps:.0f}"
                )

            # Guardar melhor modelo
            if ep_reward > best_reward:
                best_reward = ep_reward
                agent.save(os.path.join(log_dir, "best_model.pt"))

            # Checkpoint periódico
            if episode % 100 == 0:
                agent.save(os.path.join(log_dir, f"checkpoint_{agent.total_steps}.pt"))

            obs, info = env.reset()

    # Guardar modelo final
    agent.save(os.path.join(log_dir, "final_model.pt"))

    # Guardar métricas
    np.savez(
        os.path.join(log_dir, "training_metrics.npz"),
        rewards=episode_rewards,
        lengths=episode_lengths,
        cones_hit=episode_cones_hit,
        time_penalties=episode_time_penalties,
        progress=episode_progress,
    )

    env.close()
    print(f"\n[DONE] Treino completo. Modelos guardados em {log_dir}")


def train_sb3(args):
    """Treino com Stable-Baselines3 (recomendado para produção)."""
    try:
        from stable_baselines3 import SAC, PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        from stable_baselines3.common.callbacks import (
            EvalCallback, CheckpointCallback
        )
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("[ERRO] stable-baselines3 não instalado!")
        print("Instalar com: pip install stable-baselines3[extra]")
        return

    print("=" * 60)
    print(f"  Formula Student RL - Treino SB3 {args.algo.upper()}")
    print("=" * 60)

    log_dir = f"runs/sb3_{args.algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)

    # Criar ambientes vetorizados
    def make_env(rank, seed=0):
        def _init():
            env = FSRacingEnv(
                randomize_track=True,
                domain_randomization=True,
                max_episode_steps=args.max_ep_steps,
                track_seed=seed + rank,
            )
            env = Monitor(env, os.path.join(log_dir, f"monitor_{rank}"))
            return env
        return _init

    n_envs = args.n_envs
    if n_envs > 1:
        train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        train_env = DummyVecEnv([make_env(0)])

    eval_env = DummyVecEnv([make_env(99, seed=9999)])

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=10000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="fs_racing",
    )

    # Criar modelo
    if args.algo == 'sac':
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            learning_starts=5000,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            tensorboard_log=os.path.join(log_dir, "tb"),
        )
    elif args.algo == 'ppo':
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            tensorboard_log=os.path.join(log_dir, "tb"),
        )
    else:
        print(f"[ERRO] Algoritmo '{args.algo}' não suportado. Usar 'sac' ou 'ppo'.")
        return

    # Treinar
    print(f"\nA treinar {args.algo.upper()} por {args.total_steps} steps...")
    print(f"Ambientes paralelos: {n_envs}")
    print(f"Logs em: {log_dir}")
    print(f"TensorBoard: tensorboard --logdir {os.path.join(log_dir, 'tb')}\n")

    model.learn(
        total_timesteps=args.total_steps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Guardar modelo final
    model.save(os.path.join(log_dir, "final_model"))

    train_env.close()
    eval_env.close()
    print(f"\n[DONE] Treino completo. Modelos guardados em {log_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Treino RL para Formula Student Racing"
    )
    parser.add_argument('--mode', type=str, default='custom',
                        choices=['custom', 'sb3'],
                        help='Modo de treino: custom SAC ou stable-baselines3')
    parser.add_argument('--algo', type=str, default='sac',
                        choices=['sac', 'ppo'],
                        help='Algoritmo (apenas para modo sb3)')
    parser.add_argument('--total-steps', type=int, default=500_000,
                        help='Número total de steps de treino')
    parser.add_argument('--max-ep-steps', type=int, default=5000,
                        help='Máximo de steps por episódio')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Número de ambientes paralelos (sb3)')
    parser.add_argument('--render', action='store_true',
                        help='Renderizar durante treino (apenas custom)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Caminho para checkpoint (retomar treino)')

    args = parser.parse_args()

    if args.mode == 'custom':
        train_custom_sac(args)
    else:
        train_sb3(args)


if __name__ == '__main__':
    main()
