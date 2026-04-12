import gymnasium as gym
from envs.trading import TradingEnv
from graders import grade_episode
import numpy as np

def test_task(task_id):
    print(f"\n--- Testing Task: {task_id} ---")
    env = TradingEnv()
    obs, info = env.reset(options={"task_id": task_id})
    print(f"Initial State - Risk: {info['portfolio_risk']:.2f}, Vol: {info['volatility']:.2f}, Stress: {info['trader_stress']}")
    
    trajectory = []
    for _ in range(10): # Small sample
        action = 1 # SILENT
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append({
            'observation': obs.tolist(),
            'action': action,
            'reward': reward,
            'info': info
        })
        if terminated or truncated:
            break
            
    score = grade_episode(trajectory)
    print(f"Survival: {len(trajectory)} steps. Final Risk: {info['portfolio_risk']:.2f}. Grader Score: {score}")

if __name__ == "__main__":
    for task in ["stability_basic", "volatility_spike", "market_crash"]:
        test_task(task)
