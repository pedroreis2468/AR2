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
        terminate_on_cone=not args.no_terminate_on_cone,
        tracks_dir=args.tracks_dir,
        use_orange_cones=not args.legacy_obs,
    )

    # Criar agente
    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dims=(256, 256),
        lr_actor=args.lr,
        lr_critic=args.lr,
        gamma=0.99,
        tau=0.005,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        device=args.device,
    )

    print(f"[INFO] Obs dim: {env.observation_space.shape[0]} | "
          f"Action dim: {env.action_space.shape[0]} | "
          f"Learning starts: {args.learning_starts} steps")

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
    episode_progresses = []
    start_time = time.time()

    # CSV para análise offline das métricas
    csv_path = os.path.join(log_dir, "episode_metrics.csv")
    with open(csv_path, 'w') as f:
        f.write("episode,steps,reward,length,progress,speed_kmh,laps\n")

    obs, info = env.reset()

    try:
        from tqdm import tqdm
        pbar = tqdm(total=args.total_steps, initial=agent.total_steps, desc="Treino SAC")
    except ImportError:
        pbar = None
        print("[Aviso] tqdm não instalado. Use pip install tqdm para ver a barra de progresso.")

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
        if pbar:
            pbar.update(1)

        # Atualizar agente
        if agent.total_steps >= agent.learning_starts:
            metrics = agent.update()
        else:
            metrics = {}

        obs = next_obs

        if done:
            episode += 1
            ep_reward = info.get('episode_reward', 0.0)
            ep_length = info.get('step', 0)
            ep_progress = info.get('total_progress', 0.0)
            ep_laps = info.get('laps_completed', 0)
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            episode_progresses.append(ep_progress)

            # Guardar linha no CSV
            with open(csv_path, 'a') as f:
                f.write(f"{episode},{agent.total_steps},{ep_reward:.2f},"
                        f"{ep_length},{ep_progress:.1f},"
                        f"{info.get('speed_kmh', 0):.1f},{ep_laps}\n")

            # Log a cada 5 episódios
            if episode % 5 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_length = np.mean(episode_lengths[-50:])
                avg_progress = np.mean(episode_progresses[-50:])
                elapsed = time.time() - start_time
                fps = agent.total_steps / max(elapsed, 1)

                msg = (
                    f"Ep {episode:5d} | "
                    f"Steps {agent.total_steps:8d} | "
                    f"Rew {ep_reward:7.1f} | "
                    f"Avg50 {avg_reward:7.1f} | "
                    f"Prog {ep_progress:6.0f}m | "
                    f"AvgProg {avg_progress:6.0f}m | "
                    f"Len {ep_length:5d} | "
                    f"Speed {info.get('speed_kmh', 0):.1f}km/h | "
                    f"α {metrics.get('alpha', 0):.3f} | "
                    f"FPS {fps:.0f}"
                )
                if pbar:
                    pbar.write(msg)
                else:
                    print(msg)

            # Guardar melhor modelo
            if ep_reward > best_reward:
                best_reward = ep_reward
                agent.save(os.path.join(log_dir, "best_model.pt"))
                if pbar:
                    pbar.write(f"  ★ Novo melhor modelo: {ep_reward:.1f}")

            # Checkpoint periódico
            if episode % 100 == 0:
                agent.save(os.path.join(log_dir, f"checkpoint_{agent.total_steps}.pt"))

            obs, info = env.reset()

    if pbar:
        pbar.close()

    # Guardar modelo final
    agent.save(os.path.join(log_dir, "final_model.pt"))

    # Guardar métricas
    np.savez(
        os.path.join(log_dir, "training_metrics.npz"),
        rewards=episode_rewards,
        lengths=episode_lengths,
    )

    env.close()
    print(f"\n[DONE] Treino completo. Modelos guardados em {log_dir}")


def train_sb3(args):
    """Treino com Stable-Baselines3 (recomendado para produção)."""
    try:
        from stable_baselines3 import SAC, PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
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

    # Resolver subset de pistas a partir do split
    from env.track_splits import TRAIN_TRACKS, TEST_TRACKS
    if args.split == 'train':
        allowed_tracks = TRAIN_TRACKS
    elif args.split == 'test':
        allowed_tracks = TEST_TRACKS
    else:  # 'all'
        allowed_tracks = None
    print(f"[INFO] Split: {args.split} | seed: {args.seed} | "
          f"pistas: {allowed_tracks if allowed_tracks else 'all'}")

    # Nome do run (com algo, split, seed)
    if args.run_name:
        run_name = args.run_name
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"sb3_{args.algo}_{args.split}_seed{args.seed}_{ts}"
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Criar ambientes vetorizados
    def make_env(rank, seed=0):
        def _init():
            import math
            env = FSRacingEnv(
                randomize_track=True,
                domain_randomization=not args.no_dr,
                max_episode_steps=args.max_ep_steps,
                terminate_on_cone=not args.no_terminate_on_cone,
                track_seed=seed + rank,
                tracks_dir=args.tracks_dir,
                allowed_tracks=allowed_tracks,
                use_orange_cones=not args.legacy_obs,
                use_alignment_reward=not args.no_alignment,
                persistent_cone_knockdown=not args.no_persistent_knockdown,
                cone_fov=math.radians(args.cone_fov_deg),
                cone_max_range=args.cone_range,
                sensor_noise=not args.no_sensor_noise,
                max_speed_override=args.max_speed,
            )
            env = Monitor(env, os.path.join(log_dir, f"monitor_{rank}"))
            return env
        return _init

    n_envs = args.n_envs
    base_seed = args.seed * 1000  # ortogonalidade entre seeds
    if n_envs > 1:
        train_env = SubprocVecEnv([make_env(i, seed=base_seed) for i in range(n_envs)])
    else:
        train_env = DummyVecEnv([make_env(0, seed=base_seed)])

    # Eval env: usa SEMPRE o mesmo split (train) durante o treino para curvas comparáveis.
    # A avaliação final em pistas inéditas é feita por scripts/eval_per_track.py.
    eval_env = (SubprocVecEnv([make_env(99, seed=base_seed + 9999)])
                if n_envs > 1 else DummyVecEnv([make_env(99, seed=base_seed + 9999)]))

    # Normalizar recompensas — reduz a magnitude dos gradientes do actor
    # e estabiliza Q-values. Não normaliza observações (já feito no env).
    # clip_reward=50 (subido de 10) para não estrangular o lap bonus de +200.
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True,
                             clip_reward=args.clip_reward, gamma=0.99)
    eval_env  = VecNormalize(eval_env,  norm_obs=False, norm_reward=False,
                             clip_reward=args.clip_reward, gamma=0.99, training=False)

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
        # target_entropy: controla quanto exploração o SAC mantém.
        # Default SB3 = -action_dim = -2, que colapsa rápido.
        # Usar -0.5 mantém entropia alta por mais tempo.
        te = args.target_entropy  # float ou 'auto'
        model = SAC(
            "MlpPolicy",
            train_env,
            device=args.device,
            seed=args.seed,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=0.99,
            tau=0.005,
            learning_starts=args.learning_starts,
            target_entropy=te,
            ent_coef='auto',          # auto-tuning, mas guiado por target_entropy
            train_freq=1,             # update a cada step
            gradient_steps=1,
            policy_kwargs=dict(
                net_arch=[256, 256],
                optimizer_kwargs=dict(eps=1e-5),  # Adam eps mais estável
            ),
            verbose=1,
            tensorboard_log=os.path.join(log_dir, "tb"),
        )
        print(f"[SAC] target_entropy={te} | buffer={args.buffer_size} | "
              f"batch={args.batch_size} | lr={args.lr}")
    elif args.algo == 'ppo':
        model = PPO(
            "MlpPolicy",
            train_env,
            device=args.device,
            seed=args.seed,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=min(args.batch_size, 2048),
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
    # Guardar normalization stats para usar na avaliação
    train_env.save(os.path.join(log_dir, "vecnormalize.pkl"))

    train_env.close()
    eval_env.close()
    print(f"\n[DONE] Treino completo. Modelos guardados em {log_dir}")
    print(f"       VecNormalize stats: {os.path.join(log_dir, 'vecnormalize.pkl')}")


def main():
    parser = argparse.ArgumentParser(
        description="Treino RL para Formula Student Racing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Core Options ---
    core_group = parser.add_argument_group('Core Options')
    core_group.add_argument('--mode', type=str, default='custom', choices=['custom', 'sb3'],
                            help='Modo de treino: custom SAC ou stable-baselines3')
    core_group.add_argument('--algo', type=str, default='sac', choices=['sac', 'ppo'],
                            help='Algoritmo (apenas para modo sb3)')
    core_group.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                            help='Dispositivo (auto, cuda, cpu)')

    # --- Environment Options ---
    env_group = parser.add_argument_group('Environment Options')
    env_group.add_argument('--tracks-dir', type=str, default='tracks',
                           help='Diretório com pistas YAML')
    env_group.add_argument('--max-ep-steps', type=int, default=1500,
                           help='Máximo de steps por episódio (1500 evita episódios zombie)')
    env_group.add_argument('--n-envs', type=int, default=4,
                           help='Número de ambientes paralelos (sb3)')
    env_group.add_argument('--render', action='store_true',
                           help='Renderizar durante treino (apenas custom)')
    env_group.add_argument('--no-terminate-on-cone', action='store_true',
                           help='Não terminar episódio imediatamente ao bater num cone')
    env_group.add_argument('--legacy-obs', action='store_true',
                           help='Treinar modelo com observação 21 dims (ignorar cones laranjas)')

    # --- Hyperparameters ---
    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument('--total-steps', type=int, default=500_000,
                             help='Número total de steps de treino')
    hyper_group.add_argument('--lr', type=float, default=3e-4,
                             help='Learning rate (actor e critic)')
    hyper_group.add_argument('--batch-size', type=int, default=256,
                             help='Tamanho do batch para treino')
    hyper_group.add_argument('--buffer-size', type=int, default=500_000,
                             help='Tamanho do replay buffer')
    hyper_group.add_argument('--learning-starts', type=int, default=1000,
                             help='Steps iniciais aleatórios antes de o SAC começar a aprender')
    hyper_group.add_argument('--target-entropy', type=float, default=-0.5,
                             help='Entropia alvo do SAC. Default=-0.5 mantém exploração. '
                                  'SB3 default seria -action_dim=-2 (colapsa rápido)')

    # --- Checkpointing ---
    ckpt_group = parser.add_argument_group('Checkpointing')
    ckpt_group.add_argument('--checkpoint', type=str, default=None,
                            help='Caminho para checkpoint (retomar treino)')

    # --- Reproducibilidade e Splits ---
    repro_group = parser.add_argument_group('Reproducibility & Splits')
    repro_group.add_argument('--seed', type=int, default=0,
                             help='Seed para reprodutibilidade (afeta numpy, torch, env)')
    repro_group.add_argument('--split', type=str, default='train',
                             choices=['train', 'test', 'all'],
                             help='Subset de pistas: train (9), test (5) ou all (14). '
                                  'Para o treino principal usar "train"; "test" só para sanity checks.')
    repro_group.add_argument('--run-name', type=str, default=None,
                             help='Nome do run (default: auto-gerado com timestamp)')
    repro_group.add_argument('--clip-reward', type=float, default=50.0,
                             help='Clip da reward normalizada no VecNormalize. '
                                  'Default 50.0 (subido de 10.0 para não estrangular o lap bonus +200).')

    # --- Ablações ---
    abl_group = parser.add_argument_group('Ablations')
    abl_group.add_argument('--no-dr', action='store_true',
                           help='Ablação: desligar domain randomization (massa, atrito, drag, '
                                'posição/velocidade inicial, ruído do sensor).')
    abl_group.add_argument('--no-alignment', action='store_true',
                           help='Ablação: desligar termo r_alignment (v · cos(heading_err) × 0.4) '
                                'da função de recompensa.')
    abl_group.add_argument('--no-persistent-knockdown', action='store_true',
                           help='Ablação: cones derrubados continuam visíveis ao sensor '
                                '(em vez de desaparecerem persistentemente).')
    abl_group.add_argument('--cone-fov-deg', type=float, default=180.0,
                           help='FOV do sensor de cones em graus (default: 180).')
    abl_group.add_argument('--cone-range', type=float, default=15.0,
                           help='Alcance máximo do sensor de cones em metros (default: 15).')
    abl_group.add_argument('--no-sensor-noise', action='store_true',
                           help='Ablação: desligar ruído gaussiano do sensor de cones.')
    abl_group.add_argument('--max-speed', type=float, default=None,
                           help='Sobrescrever max_speed do veículo em m/s (default: usa o do VehicleParams).')

    args = parser.parse_args()

    # Aplicar seed globalmente
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    if args.mode == 'custom':
        train_custom_sac(args)
    else:
        train_sb3(args)


if __name__ == '__main__':
    main()
