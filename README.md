---
title: Stock Market Monitoring Assistant
emoji: none
colorFrom: yellow
colorTo: red
sdk: docker
app_file: app.py
pinned: false
---

# Stock Market Monitoring Assistant

An AI-powered Stock Market Monitoring Terminal designed to combat alert fatigue in high-frequency trading environments. This space uses Reinforcement Learning (PPO) to intelligently decide when to alert a trader about market risks vs. when to intervene or remain silent.

### Features
- **Real-time Telemetry**: Plotly-based tracking of Portfolio Risk and Trader Stress.
- **Cognitive Load Modeling**: RL agent trained to balance safety with trader fatigue.
- **Bloomber-Edition Aesthetic**: High-contrast interface for professional risk analysis.
- **Multi-Level Tasks**: Integrated scenarios for Base Stability, Volatility Spikes, and Market Crashes.

### Technical Specification

#### Observation Space (Box(5))
| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | Portfolio Risk | [0, 1] | Mean risk across all asset classes (0.8+ is critical). |
| 1 | Market Volatility | [0, 1] | Current market turbulence level. |
| 2 | Error Signal | [0, 1] | EMA of stress-induced decision errors. |
| 3 | Time Pressure | [0, 1] | Episode progress (Steps / Max Steps). |
| 4 | Last Action | [0, 1] | Normalized action ID of the previous step. |

#### Action Space (Discrete(3))
- **0: ALERT**: Issue warning. Reduces risk but increases noise and trader stress.
- **1: SILENT**: Monitor only. Natural risk evolution based on volatility.
- **2: INTERVENE**: Manual override. High risk reduction, but extreme stress cost.

### Task Scenarios
- **stability_basic**: Easy start for baseline validation. Maintains risk below 0.5.
- **volatility_spike**: Medium difficulty. Responds to market turbulence (0.8 volatility).
- **market_crash**: Hard scenario. Recovers from 0.8+ risk and maximum stress state.

### Local Development
```bash
pip install -r requirements.txt
python app.py
```
