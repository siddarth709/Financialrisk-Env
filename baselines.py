import numpy as np

class BaselineAgent:
    def __init__(self, action):
        self.action = action

    def predict(self, obs, deterministic=True):
        return self.action, {}

class AlwaysAlertAgent(BaselineAgent):
    def __init__(self):
        super().__init__(0)

class NeverAlertAgent(BaselineAgent):
    def __init__(self):
        super().__init__(1)

class AlwaysInterveneAgent(BaselineAgent):
    def __init__(self):
        super().__init__(2)

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        return self.action_space.sample(), {}

def get_baseline_agent(agent_name, env):
    if agent_name == "Always Alert":
        return AlwaysAlertAgent()
    elif agent_name == "Never Alert":
        return NeverAlertAgent()
    elif agent_name == "Always Intervene":
        return AlwaysInterveneAgent()
    elif agent_name == "Random":
        return RandomAgent(env.action_space)
    else:
        raise ValueError(f"Unknown baseline: {agent_name}")
