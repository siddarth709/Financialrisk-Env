import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    def __init__(self, max_steps=100, render_mode=None):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.step_count = 0
        self.portfolio_risk = 0.0
        self.market_volatility = 0.0
        self.trader_stress = 1.0
        self.last_action = 1
        self.error_signal = 0.0
        self.error_ema_alpha = 0.3
        self.share_names = ["Tech", "Health", "Energy", "Finance"]
        self.num_shares = len(self.share_names)
        self.share_risks = np.zeros(self.num_shares)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.share_risks = self.np_random.uniform(0.1, 0.3, size=self.num_shares)
        self.portfolio_risk = float(np.mean(self.share_risks))
        self.market_volatility = float(self.np_random.uniform(0.2, 0.5))
        self.trader_stress = float(self.np_random.choice([0.0, 1.0, 2.0], p=[0.3, 0.4, 0.3]))
        self.last_action = 1
        self.error_signal = 0.0

        obs = self._get_obs()
        info = {
            "trader_stress": int(round(self.trader_stress)),
            "portfolio_risk": float(self.portfolio_risk),
            "volatility": float(self.market_volatility),
            "step": int(self.step_count),
            "share_risks": {name: float(risk) for name, risk in zip(self.share_names, self.share_risks)}
        }
        return obs, info

    def _get_obs(self):
        time_pressure = min(1.0, float(self.step_count) / self.max_steps)
        return np.array(
            [
                float(self.portfolio_risk),
                float(self.market_volatility),
                float(self.error_signal),
                float(time_pressure),
                float(self.last_action) / 2.0
            ], 
            dtype=np.float32,
        )
    
    def step(self, action):
        assert self.action_space.contains(action)
        self.step_count += 1
        
        action = int(action)
        shock_event = self.np_random.random() < 0.05
        vol_change = self.np_random.normal(0.02 if shock_event else 0.0, 0.05)
        self.market_volatility = np.clip(self.market_volatility + vol_change, 0.0, 1.0)
        
        stress_noise = self.np_random.normal(0, 0.1)
        disruption = 0.0
        if action == 0 and self.portfolio_risk < 0.3:
            disruption = 0.2
        
        self.trader_stress = np.clip(self.trader_stress + stress_noise + (self.market_volatility * 0.1) + disruption, 0.0, 2.0)
        stress_level = int(round(self.trader_stress))
        
        risk_increase = self.market_volatility * (0.05 + 0.05 * stress_level)
        share_noise = self.np_random.normal(0, 0.02, size=self.num_shares)
        self.share_risks = np.clip(self.share_risks + risk_increase + share_noise, 0.0, 1.0)
        self.portfolio_risk = float(np.mean(self.share_risks))
        
        error_prob = {0: 0.02, 1: 0.08, 2: 0.2}[stress_level]
        error_event = self.np_random.choice([0, 1], p=[1-error_prob, error_prob])
        self.error_signal = self.error_ema_alpha * error_event + (1 - self.error_ema_alpha) * self.error_signal

        reward = -0.1
        if action == 0:
            if self.portfolio_risk > 0.6:
                reward += 3.0
                self.share_risks *= 0.8
                self.portfolio_risk = float(np.mean(self.share_risks))
            elif self.portfolio_risk < 0.3:
                reward -= 5.0
            else:
                reward -= 2.0
        elif action == 2:
            if self.portfolio_risk > 0.8:
                reward += 5.0
                self.share_risks *= 0.5
                self.portfolio_risk = float(np.mean(self.share_risks))
            else:
                reward -= 5.0

        if self.portfolio_risk >= 1.0:
            reward -= 10.0

        terminated = self.portfolio_risk <= 0.1 or self.portfolio_risk >= 1.0
        if self.portfolio_risk <= 0.1:
            reward += 10.0

        truncated = self.step_count >= self.max_steps
        self.last_action = action
        
        obs = self._get_obs()
        info = {
            "portfolio_risk": float(self.portfolio_risk),
            "volatility": float(self.market_volatility),
            "trader_stress": int(stress_level),
            "step": int(self.step_count),
            "action_meaning": ["ALERT", "SILENT", "INTERVENE"][action],
            "share_risks": {name: float(risk) for name, risk in zip(self.share_names, self.share_risks)}
        }

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        pass

    def close(self):
        pass