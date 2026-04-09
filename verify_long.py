import gymnasium as gym
from envs.trading import TradingEnv
from envs.cognitive_load_env import CognitiveLoadEnv
import numpy as np

def test_envs_long():
    print("Testing TradingEnv for 50 steps...")
    trading_env = TradingEnv()
    obs, info = trading_env.reset()
    for i in range(50):
        action = trading_env.action_space.sample()
        obs, reward, terminated, truncated, info = trading_env.step(action)
        print(f"Step {i}: Reward={reward:.2f}, Risk={info['portfolio_risk']:.2f}, Stress={info['trader_stress']}")
        if terminated or truncated:
            print("Finished TradingEnv early")
            break
    
    print("\nTesting CognitiveLoadEnv for 50 steps...")
    cognitive_env = CognitiveLoadEnv()
    obs, info = cognitive_env.reset()
    for i in range(50):
        action = cognitive_env.action_space.sample()
        obs, reward, terminated, truncated, info = cognitive_env.step(action)
        print(f"Step {i}: Reward={reward:.2f}, Load={obs[0]:.2f}")
        if terminated or truncated:
            print("Finished CognitiveLoadEnv early")
            break
    
    print("\nLong verification successful!")

if __name__ == "__main__":
    test_envs_long()
