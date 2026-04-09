import os
import gymnasium as gym
from stable_baselines3 import PPO
from envs.trading import TradingEnv
from baselines import get_baseline_agent
import numpy as np
import pandas as pd

def run_episode(env, agent):
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    alerts = 0
    interventions = 0
    
    while not (done or truncated):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if action == 0: alerts += 1
        if action == 2: interventions += 1
        
    return {
        "reward": total_reward,
        "steps": steps,
        "alerts": alerts,
        "interventions": interventions,
        "risk_final": info["portfolio_risk"]
    }

def compare_agents(num_episodes=20):
    env = TradingEnv()
    agents = {
        "PPO Agent": None,
        "Always Alert": get_baseline_agent("Always Alert", env),
        "Never Alert": get_baseline_agent("Never Alert", env),
        "Random": get_baseline_agent("Random", env)
    }
    
    model_path = "./models/ppo_trading_final"
    if os.path.exists(model_path + ".zip"):
        agents["PPO Agent"] = PPO.load(model_path)
    else:
        print(f"Warning: Trained PPO model not found at {model_path}. Skipping PPO.")
        del agents["PPO Agent"]

    results = []
    
    print(f"Starting comparison of {len(agents)} agents over {num_episodes} episodes each...")
    
    for name, agent in agents.items():
        ep_rewards = []
        ep_alerts = []
        ep_fails = 0
        
        for _ in range(num_episodes):
            res = run_episode(env, agent)
            ep_rewards.append(res["reward"])
            ep_alerts.append(res["alerts"])
            if res["risk_final"] >= 0.99:
                ep_fails += 1
        
        results.append({
            "Agent": name,
            "Avg Reward": np.mean(ep_rewards),
            "Avg Alerts": np.mean(ep_alerts),
            "Failure Rate": ep_fails / num_episodes
        })
    
    df = pd.DataFrame(results)
    print("\n--- COMPARISON RESULTS ---")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    compare_agents()
