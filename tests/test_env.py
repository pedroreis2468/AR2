"""Testes do ambiente Gymnasium (env/racing_env.py) e infra de pistas."""
import numpy as np

from conftest import TRACKS_DIR
from env.racing_env import FSRacingEnv
from env.track_splits import TRAIN_TRACKS, TEST_TRACKS
from env.track_loader import YAMLTrackLoader

_INFO_KEYS = ('speed_kmh', 'laps_completed', 'cones_hit', 'total_progress')


# ─── Splits canonicos ────────────────────────────────────────────────────
def test_splits_sizes_and_disjoint():
    assert len(TRAIN_TRACKS) == 9
    assert len(TEST_TRACKS) == 5
    assert set(TRAIN_TRACKS).isdisjoint(set(TEST_TRACKS))


def test_track_loader_has_all_split_tracks():
    loader = YAMLTrackLoader(TRACKS_DIR)
    names = set(loader.track_names)
    missing = [t for t in TRAIN_TRACKS + TEST_TRACKS if t not in names]
    assert not missing, f"pistas em falta no loader: {missing}"


# ─── Espacos de accao/observacao ─────────────────────────────────────────
def test_action_space(env):
    assert env.action_space.shape == (2,)
    assert np.allclose(env.action_space.low, -1.0)
    assert np.allclose(env.action_space.high, 1.0)


def test_observation_space_dim_default(env):
    assert env.observation_space.shape == (24,)


def test_observation_space_dim_legacy():
    e = FSRacingEnv(tracks_dir=TRACKS_DIR, track_name=TRAIN_TRACKS[0],
                    randomize_track=False, domain_randomization=False,
                    use_orange_cones=False)
    try:
        assert e.observation_space.shape == (18,)
    finally:
        e.close()


# ─── Contrato Gymnasium: reset / step ────────────────────────────────────
def test_reset_contract(env):
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (24,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    for k in _INFO_KEYS:
        assert k in info, f"chave '{k}' em falta no info"


def test_step_contract(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (24,)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert np.isfinite(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(info, dict)


def test_episode_runs_without_crash(env):
    env.reset(seed=0)
    steps = 0
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        steps += 1
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    assert steps > 0


def test_truncation_at_max_steps():
    """Parado (accao nula), o episodio acaba por truncar no limite de steps."""
    e = FSRacingEnv(tracks_dir=TRACKS_DIR, track_name=TRAIN_TRACKS[0],
                    randomize_track=False, domain_randomization=False,
                    max_episode_steps=50)
    try:
        e.reset(seed=0)
        terminated = truncated = False
        for _ in range(60):
            _, _, terminated, truncated, _ = e.step(np.array([0.0, 0.0], dtype=np.float32))
            if terminated or truncated:
                break
        assert terminated or truncated
    finally:
        e.close()
