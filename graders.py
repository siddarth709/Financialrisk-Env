import numpy as np

def grade_episode(trajectory):
    """
    Trajectory is a list of dicts: [{'observation': ..., 'action': ..., 'reward': ..., 'info': ...}]
    Returns a score between 0.0 and 1.0.
    """
    if not trajectory:
        return 0.01

    # 1. Survival Rate (40%)
    max_steps = 100 # Default max steps
    steps_survived = len(trajectory)
    survival_score = min(1.0, steps_survived / max_steps)

    # 2. Risk Mitigation Efficiency (40%)
    # Average portfolio risk across the episode. Goal is to keep it low.
    risks = [t['info'].get('portfolio_risk', 1.0) for t in trajectory]
    avg_risk = np.mean(risks)
    # Score 1.0 if avg_risk <= 0.3, score 0.0 if avg_risk >= 0.9
    risk_score = np.clip((0.9 - avg_risk) / 0.6, 0.0, 1.0)

    # 3. Trader Stress Management (20%)
    # High stress level (2) leads to errors. Goal is to minimize time in stress level 2.
    stress_levels = [t['info'].get('trader_stress', 0) for t in trajectory]
    stress_2_ratio = stress_levels.count(2) / len(stress_levels)
    stress_score = 1.0 - stress_2_ratio

    final_score = (survival_score * 0.4) + (risk_score * 0.4) + (stress_score * 0.2)
    # Ensure score is strictly between 0 and 1 as per requirement
    final_score = 0.01 + 0.98 * final_score
    return float(np.round(final_score, 3))

if __name__ == "__main__":
    # Example usage for validator
    import sys
    import json
    
    # If the validator pipes a JSON trajectory
    try:
        data = json.load(sys.stdin)
        score = grade_episode(data)
        print(f"SCORE: {score}")
    except:
        pass
