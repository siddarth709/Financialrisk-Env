import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.trading import TradingEnv

def train():
    LOG_DIR = "./logs/"
    MODEL_DIR = "./models/"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = TradingEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=MODEL_DIR,
        name_prefix="ppo_trading"
    )

    print("Starting training on Bloomberg-style Stock Market Assistant...")
    total_timesteps = 100000
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="PPO_StockMarket_Assistant"
    )

    model_path = os.path.join(MODEL_DIR, "ppo_trading_final")
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}")

if __name__ == "__main__":
    train()
