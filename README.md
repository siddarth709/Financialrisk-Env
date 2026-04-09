---
title: Stock Market Monitoring Assistant
emoji: 📊
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
- **Smart Alerting**: RL agent trained to balance safety with cognitive bandwidth.
- **Bloomberg Aesthetic**: High-contrast, data-dense interface for professional risk analysis.

### Local Development
```bash
pip install -r requirements.txt
python app.py
```
