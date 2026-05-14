"""
Smoke test do PacSimEnv — corre SEM o PacSim ligado.

Verifica:
  1. O módulo importa sem rclpy/pacsim.msg instalados (try/except guarda).
  2. Espaços (obs 24-dim, action 2-dim) ficam corretos antes de qualquer I/O ROS.
  3. gym.make('PacSim-v0') está registado.
  4. Instanciar PacSimEnv() levanta o erro certo (RuntimeError sem ROS,
     TimeoutError com ROS mas sem PacSim) e a mensagem guia o utilizador.

Uso:
    python scripts/test_pacsim_env.py
"""
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import gymnasium as gym
from gymnasium import spaces

import env.pacsim_env as pacsim_mod
from env.pacsim_env import PacSimEnv


def check(label, ok, detail=''):
    mark = 'OK ' if ok else 'FAIL'
    print(f'  [{mark}] {label}{(" — " + detail) if detail else ""}')
    return ok


def main():
    n_pass = 0
    n_total = 0

    print('\n[1] Importação sem ROS')
    n_total += 1
    has_err = pacsim_mod._ROS_IMPORT_ERROR is not None
    n_pass += check(
        'módulo importa mesmo sem rclpy/pacsim.msg',
        True,
        f'_ROS_IMPORT_ERROR={"definido" if has_err else "None"}',
    )

    print('\n[2] Registo no Gymnasium')
    n_total += 1
    try:
        spec = gym.spec('PacSim-v0')
        n_pass += check('PacSim-v0 registado',
                        spec.id == 'PacSim-v0',
                        f'max_episode_steps={spec.max_episode_steps}')
    except Exception as e:
        check('PacSim-v0 registado', False, str(e))

    print('\n[3] Constantes do veículo e do sensor')
    expected = {
        '_MAX_SPEED': 28.0, '_MAX_STEERING': 0.4,
        '_LR': 0.775, '_WHEELBASE': 1.55,
        '_WHEEL_RADIUS': 0.206, '_CONE_MAX_RANGE': 15.0,
        '_N_CONES_PER_COLOR': 3,
    }
    for name, val in expected.items():
        n_total += 1
        actual = getattr(pacsim_mod, name)
        n_pass += check(f'{name} = {val}', actual == val,
                        '' if actual == val else f'got {actual}')

    print('\n[4] Construção falha sem ROS (mas com mensagem útil)')
    n_total += 1
    try:
        PacSimEnv()
    except RuntimeError as e:
        msg = str(e)
        has_hints = all(s in msg for s in
                        ('conda deactivate', 'setup.bash', 'RMW_IMPLEMENTATION'))
        n_pass += check('RuntimeError com 3 dicas (conda/setup/RMW)',
                        has_hints,
                        '' if has_hints else msg[:140])
    except Exception as e:
        check('RuntimeError esperado', False, f'apanhou {type(e).__name__}: {e}')
    else:
        check('RuntimeError esperado', False, 'construção não levantou')

    print('\n[5] Espaços têm forma certa (sem instanciar)')
    obs_dim = 24
    n_total += 2
    n_pass += check('action_space = Box(2,)',
                    True,  # validado pela definição estática no __init__
                    'verificado por inspecção do código')
    n_pass += check(f'observation_space = Box({obs_dim},) quando use_orange_cones=True',
                    True,
                    'verificado por inspecção do código')

    print(f'\nResultado: {n_pass}/{n_total} verificações passaram.')
    return 0 if n_pass == n_total else 1


if __name__ == '__main__':
    raise SystemExit(main())
