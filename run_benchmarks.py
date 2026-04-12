import gymnasium as gym
from envs.trading import TradingEnv
from baselines import get_baseline_agent
from graders import grade_episode
import numpy as np
import pandas as pd

def run_benchmark(agent_name, task_id, num_episodes=5):
    env = TradingEnv()
    scores = []
    
    for _ in range(num_episodes):
        obs, info = env.reset(options={"task_id": task_id})
        agent = get_baseline_agent(agent_name, env)
        
        trajectory = []
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            trajectory.append({
                'observation': obs.tolist(),
                'action': int(action),
                'reward': reward,
                'info': info
            })
            
        score = grade_episode(trajectory)
        scores.append(score)
        
    return np.mean(scores)

if __name__ == "__main__":
    tasks = ["stability_basic", "volatility_spike", "market_crash"]
    agents = ["Always Alert", "Never Alert", "Random"]
    
    results = []
    for task in tasks:
        print(f"Benchmarking Task: {task}")
        for agent in agents:
            avg_score = run_benchmark(agent, task)
            results.append({
                "Task": task,
                "Agent": agent,
                "Avg Score": avg_score
            })
            
    df = pd.DataFrame(results)
    print("\n--- BENCHMARK RESULTS ---")
    print(df.pivot(index='Task', columns='Agent', values='Avg Score'))
    df.to_json("benchmark_data.json", orient="records")
