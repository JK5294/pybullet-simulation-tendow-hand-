"""Demo: train PPO on the tendon hand environment.

Usage:
    python examples/demo_rl_train.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gymnasium as gym

# Register the environment
import tendon_hand.env.gym_env  # noqa: F401


def main():
    print("=" * 60)
    print("Tendon Hand RL — PPO Training Demo")
    print("=" * 60)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("❌ stable-baselines3 not installed. Run: pip install stable-baselines3")
        return

    env = gym.make("TendonHand-v1")
    vec_env = DummyVecEnv([lambda: env])

    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4)

    print("\n👉 Training for 10,000 steps...")
    model.learn(total_timesteps=10_000)

    model_path = "models/ppo_tendon_hand"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    print(f"\n✅ Model saved to {model_path}")

    # Quick eval
    print("\n👉 Evaluating...")
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"   Total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
