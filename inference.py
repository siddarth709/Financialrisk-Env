import os
import json
import time
import numpy as np
from openai import OpenAI
from stable_baselines3 import PPO
from envs.trading import TradingEnv

def log_event(marker, data):
    print(f"{marker} {json.dumps(data, default=lambda x: float(x) if isinstance(x, np.floating) else x)}")

class OpenEnvInference:
    def __init__(self):
        self.api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4")
        self.hf_token = os.getenv("HF_TOKEN", "")
        
        self.client = OpenAI(base_url=self.api_base_url, api_key=self.hf_token)
        self.env = TradingEnv()
        import os
        os.makedirs("./models", exist_ok=True)
        if not os.path.exists("./models/ppo_trading_final.zip"):
            try:
                from weights import WEIGHTS_B64
                import base64
                with open("./models/ppo_trading_final.zip", "wb") as f:
                    f.write(base64.b64decode(WEIGHTS_B64))
            except ImportError:
                pass
        self.model = PPO.load("./models/ppo_trading_final")
        
        log_event("[START]", {
            "model_name": self.model_name,
            "api_base": self.api_base_url,
            "timestamp": time.time(),
            "status": "initialized"
        })

    def get_rationale(self, obs, action, reward):
        try:
            prompt = f"Risk State: {obs.tolist()}. Action Taken: {action}. Immediate Reward: {reward}. Provide a brief stock market rationale for this risk mitigation step."
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Rationale unavailable: {str(e)}"

    def run(self, max_steps=20):
        obs, info = self.env.reset()
        total_reward = 0
        
        for i in range(max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(action)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            rationale = self.get_rationale(obs, action, reward)
            
            log_event("[STEP]", {
                "step": i + 1,
                "observation": obs.tolist(),
                "action": action,
                "action_label": ["ALERT", "SILENT", "INTERVENE"][action],
                "reward": float(reward),
                "rationale": rationale,
                "portfolio_risk": float(info.get("portfolio_risk", 0)),
                "trader_stress": int(info.get("trader_stress", 0))
            })
            
            obs = next_obs
            if terminated or truncated:
                break
        
        log_event("[END]", {
            "total_reward": float(total_reward),
            "final_step": i + 1,
            "status": "completed"
        })

if __name__ == "__main__":
    agent = OpenEnvInference()
    agent.run()
