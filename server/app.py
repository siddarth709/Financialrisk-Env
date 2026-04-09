import os
import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3 import PPO
from envs.trading import TradingEnv

env = TradingEnv()
model = None
if os.path.exists("./models/ppo_trading_final.zip"):
    model = PPO.load("./models/ppo_trading_final")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/reset")
async def reset():
    obs, info = env.reset()
    return {"observation": obs.tolist(), "info": info}

@app.post("/step")
async def step(request: Request):
    data = await request.json()
    action = data.get("action", 1)
    obs, reward, terminated, truncated, info = env.step(int(action))
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    }

@app.get("/state")
async def state():
    obs = env._get_obs()
    return {
        "observation": obs.tolist(),
        "portfolio_risk": float(env.portfolio_risk),
        "trader_stress": int(round(env.trader_stress)),
        "step_count": int(env.step_count)
    }

def run_simulation(steps):
    obs, info = env.reset()
    history = []
    
    for _ in range(int(steps)):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        
        obs, reward, done, truncated, info = env.step(int(action))
        hist_entry = {
            "Step": env.step_count,
            "Risk": info["portfolio_risk"],
            "Stress": info["trader_stress"],
            "Action": info["action_meaning"]
        }
        for s_name, s_risk in info.get("share_risks", {}).items():
            hist_entry[f"Share_{s_name}"] = s_risk
        history.append(hist_entry)
        if done or truncated:
            break
            
    df = pd.DataFrame(history)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Step"], y=df["Risk"], name="Aggregate Risk", line=dict(color="#00ff00", width=3)))
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99']
    for idx, k in enumerate([col for col in df.columns if col.startswith('Share_')]):
        fig.add_trace(go.Scatter(x=df["Step"], y=df[k], name=f"{k.split('_')[1]} Share", line=dict(color=colors[idx % len(colors)], width=1, dash='dash')))
    fig.add_trace(go.Scatter(x=df["Step"], y=df["Stress"], name="Trader Stress", line=dict(color="#ff0000", width=2, dash="dot")))
    
    fig.update_layout(
        template="plotly_dark",
        title="Real-time Risk Telemetry",
        xaxis_title="Simulation Step",
        yaxis_title="Normalized Level",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0")
    )
    
    status = "⚠️ CRITICAL RISK" if env.portfolio_risk > 0.8 else "✅ STABLE"
    return fig, df, f"SIMULATION STATUS: {status} | FINAL RISK: {env.portfolio_risk:.2f}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Stock Market Monitoring Terminal")
    
    with gr.Row():
        with gr.Column(scale=2):
            plot = gr.Plot(label="Telemetry")
        with gr.Column(scale=1):
            status_text = gr.Textbox(label="System Status", interactive=False)
            steps_slider = gr.Slider(minimum=10, maximum=100, value=50, step=10, label="Simulation Horizon")
            run_btn = gr.Button("🚀 START MONITORING", variant="primary")
            
    with gr.Row():
        logs = gr.Dataframe(label="Real-time Event Log")
        
    run_btn.click(run_simulation, inputs=[steps_slider], outputs=[plot, logs, status_text])

app = gr.mount_gradio_app(app, demo, path="/")

def main():
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="Start the Stock Market Monitoring Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
