import gymnasium as gym
from envs.trading import TradingEnv
from envs.cognitive_load_env import CognitiveLoadEnv

def test_envs():
    print("Testing TradingEnv...")
    trading_env = TradingEnv()
    obs, info = trading_env.reset()
    print(f"Trading reset obs: {obs}")
    obs, reward, terminated, truncated, info = trading_env.step(1)
    print(f"Trading step reward: {reward}")
    
    print("\nTesting CognitiveLoadEnv...")
    cognitive_env = CognitiveLoadEnv()
    obs, info = cognitive_env.reset()
    print(f"Cognitive reset obs: {obs}")
    obs, reward, terminated, truncated, info = cognitive_env.step(1)
    print(f"Cognitive step reward: {reward}")
    
    print("\nVerification successful!")

if __name__ == "__main__":
    test_envs()
