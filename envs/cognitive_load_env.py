import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CognitiveLoadEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.state = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 1.0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([float(self.state)], dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action)
        action = int(action)
        
        if action == 1:
            self.state -= 0.05
        else:
            self.state += 0.1
            
        self.state = float(np.clip(self.state, 0.0, 1.0))
        reward = float(self.state if self.state > 0.2 else -1.0)
        
        terminated = bool(self.state <= 0.0)
        truncated = False
            
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass
