"""Integration tests for the Gymnasium environment."""

from __future__ import annotations

import os

import numpy as np
import pytest

import gymnasium as gym

import tendon_hand.env.gym_env  # noqa: F401

# Path to local URDF for testing (user must provide their own)
_URDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "urdf", "V3.urdf"
)


@pytest.fixture
def has_urdf():
    return os.path.isfile(_URDF_PATH)


class TestGymEnv:
    def test_env_registration(self, has_urdf):
        if not has_urdf:
            pytest.skip("No URDF found — provide your own to run simulation tests")
        env = gym.make("TendonHand-v1", urdf_path=_URDF_PATH)
        assert env is not None
        env.close()

    def test_reset(self, has_urdf):
        if not has_urdf:
            pytest.skip("No URDF found")
        env = gym.make("TendonHand-v1", urdf_path=_URDF_PATH)
        obs, info = env.reset(seed=42)
        assert obs.shape == (47,)
        assert np.all(np.isfinite(obs))
        env.close()

    def test_step(self, has_urdf):
        if not has_urdf:
            pytest.skip("No URDF found")
        env = gym.make("TendonHand-v1", urdf_path=_URDF_PATH)
        obs, _ = env.reset(seed=42)
        action = np.zeros(17, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (47,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "phase" in info
        env.close()

    def test_action_clipping(self, has_urdf):
        if not has_urdf:
            pytest.skip("No URDF found")
        env = gym.make("TendonHand-v1", urdf_path=_URDF_PATH)
        env.reset(seed=42)
        # Extreme action should still produce finite observation
        action = np.ones(17, dtype=np.float32) * 2.0
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs))
        env.close()

    def test_episode_length(self, has_urdf):
        if not has_urdf:
            pytest.skip("No URDF found")
        env = gym.make("TendonHand-v1", urdf_path=_URDF_PATH, max_steps=10)
        env.reset(seed=42)
        truncated = False
        for _ in range(15):
            if truncated:
                break
            _, _, _, truncated, _ = env.step(env.action_space.sample())
        assert truncated
        env.close()
