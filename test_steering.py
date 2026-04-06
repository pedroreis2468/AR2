import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import numpy as np
from stable_baselines3 import SAC
from env.racing_env import FSRacingEnv

model = SAC.load('runs/sb3_sac_20260403_040429/best/best_model.zip', device='cpu')

def test_env(domain_rand):
    env = FSRacingEnv(render_mode=None, terminate_on_cone=False, domain_randomization=domain_rand, tracks_dir='../pistas/tracks')
    obs, _ = env.reset()
    actions = []
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action[0])
        obs, _, done, _, _ = env.step(action)
    print(f'Steering (Noise {domain_rand}):', [round(float(a), 3) for a in actions[:10]])

test_env(False)
test_env(True)
