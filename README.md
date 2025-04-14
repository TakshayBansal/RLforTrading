# Deep Reinforcement Learning for Stock Trading

### DA221M Minor Course Project

**Team Members:**

- Takshay Bansal (230102096)
- Aditya Jain (230108002)
- Nilarnab Sutradhar (230123041)
- Nisant Sarma (230121038)

This project explores the application of **Deep Reinforcement Learning (DRL)** techniques for developing intelligent trading strategies in financial markets, forming part of the course **DA221M - Artificial Intelligence**.

---

## Prerequisites & Installation

Ensure the following packages are installed:

```bash
pip install numpy pandas matplotlib yfinance gymnasium stable-baselines3
```

Recommended environment: Python 3.8+ with Jupyter Notebook.

---

## Project Overview

We developed and evaluated multiple DRL-based agents to make trading decisions on stock price data. Agents were trained in a custom OpenAI Gym-compatible environment and assessed using key financial metrics like **net worth**, **returns**, and **Sharpe Ratio**.

We implemented and compared the following DRL algorithms:

- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- DDPG (Deep Deterministic Policy Gradient)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed Deep Deterministic Policy Gradient)

---

## Relevance to the Paper

We referenced the paper:  
**"Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy"**

This guided our architectural and experimental choices, specifically:

- Use of critic-only and actor-critic architectures
- Discrete action space: Buy, Sell, Hold
- State space comprising technical indicators like MACD and RSI
- Net worth-based reward structure aligned with the paper's reviewed strategies
- Single-asset, low-frequency backtesting scenario

---

## Implementation Summary

### 1. Data Handling

Stock data (e.g., Apple - AAPL) was retrieved from **Yahoo Finance** using `yfinance`. The dataset spans several years and includes OHLCV fields.

### 2. Technical Indicators

Computed and included in the state:

- Moving Averages (MA10, MA50)
- MACD (trend-following)
- RSI (momentum-based)

These indicators help the agent contextualize market conditions.

### 3. Custom Trading Environment

We implemented a custom Gym environment:

- Tracks cash, number of shares, and net worth
- Allows three actions: Buy (1 share), Sell (1 share), Hold
- Reward = Δ(Net Worth) − Transaction Cost (0.1% per transaction)

### 4. Agent Training

Agents were trained using **Stable-Baselines3** for ~5000 timesteps. Vectorized environments via `DummyVecEnv` supported batch training. Each model was evaluated over test data post-training.

### 5. Evaluation

Each agent's net worth trajectory was plotted and compared.  
Additionally, **Sharpe Ratios**, **average returns**, and **standard deviations** were computed using:

```python
daily_returns = np.diff(net_worths) / net_worths[:-1]
sharpe_ratio = (avg_return - risk_free_rate) / std_return
```

This enabled a risk-adjusted comparison of performance.

---

## Results

The performance ranking (based on Sharpe Ratio) was:

1. **DDPG** - Best-performing agent with the highest return and Sharpe Ratio
2. **SAC** - Strong and stable performance
3. **TD3** - Moderate risk-adjusted performance
4. **A2C** - Marginal improvements over baseline
5. **PPO** - Decreasing portfolio value over time

![Net Worth Chart](./net_worth_plot.png)

This aligns with paper insights that **actor-critic methods** like DDPG and SAC perform better in continuous environments.

---

## Limitations

- Fixed 1-share trading per step
- No portfolio diversification (single asset)
- No transaction slippage or real-world latency
- No live deployment; entirely backtested
- Training duration and depth limited by compute (5000 timesteps only)

---

## Learnings

- Developed end-to-end RL trading systems
- Understood action-reward dynamics in market simulation
- Gained experience with multiple RL algorithms and vectorized training
- Interpreted financial metrics to assess model performance
- Bridged theory from DRL literature to practical implementation

---

## Files

- `RL_Trading.ipynb` - Full implementation
- `drltrading.pdf` - Reference research paper
