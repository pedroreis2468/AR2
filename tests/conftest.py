"""Configuracao partilhada dos testes pytest.

Coloca a raiz do projeto no sys.path (para importar `env.*`) e define a
fixture `env`: um ambiente determinístico (sem domain randomization) numa
pista de treino fixa, adequado para asserts reprodutíveis.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TRACKS_DIR = os.path.join(ROOT, 'tracks')

from env.racing_env import FSRacingEnv  # noqa: E402
from env.track_splits import TRAIN_TRACKS  # noqa: E402


@pytest.fixture
def env():
    e = FSRacingEnv(
        render_mode=None,
        randomize_track=False,
        domain_randomization=False,
        tracks_dir=TRACKS_DIR,
        track_name=TRAIN_TRACKS[0],
        max_episode_steps=200,
    )
    yield e
    e.close()
