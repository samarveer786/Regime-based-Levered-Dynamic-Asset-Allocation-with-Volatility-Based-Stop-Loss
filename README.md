# Levered-Dynamic-Asset-Allocation-with-Volatility-Based-Stop-Loss

Python implementation of an institutional-grade **DeFi strategy** that combines:

- Trend-based **dynamic leverage** on ETH–stETH loops,
- Systematic **asset allocation** between ETH staking and PT stablecoins,
- **Volatility-driven de-risking** and health-factor monitoring,

with the goal of outperforming Buy & Hold ETH on a **risk-adjusted** basis.

---

## 1. Project Overview

This repository contains:

- A full **backtest engine** for ETH from Jan-2020 to 7 Oct 2025 (daily data).
- Implementation of:
  - Buy & Hold ETH benchmark
  - Fixed 2× ETH loop
  - Dynamic LTV model
  - **Levered Dynamic Asset Allocation model** (main strategy)
- Detailed performance metrics:
  - Sharpe, Sortino, Calmar, CAGR
  - Max drawdown
  - Final equity and ROI
- APY simulation stats for the loop (staking vs borrow APYs).

---

## 2. Strategy Architecture

### 2.1 Dynamic Leverage Engine

Trend detection using EMAs:

- `EMA_8`, `EMA_20`, `EMA_50` computed on daily ETH price.
- Leverage rules (`get_dyn_lev`):

  - Strong uptrend (`Price > EMA_50` and `EMA_8 > EMA_20`) → **2×**
  - Mild uptrend (`Price > EMA_50`) → **1.5×**
  - Early trend (`EMA_8 > EMA_20`) → **1×**
  - Else → **0×** (no leverage)

- Stop-loss overlay:
  - `Price < 0.95 * EMA_50` → **force 0×**
  - `Price < EMA_50` → cap leverage at **0.5×**

- **Execution:**  
  `dyn_lev_signal = dyn_lev.shift(7)` → leverage is **rebalanced weekly** based on the previous week’s trend.

### 2.2 Dynamic Asset Allocation Engine

Capital split:

- **10% permanent liquidity buffer (`w_liquid`)**.
- **90% “risk capital”**:

  - ETH staking bucket (LST)
  - PT stablecoin bucket

Staking weight is a function of 30-day ETH performance:

1. 30-day return: `eth_ret_30d`.
2. Clip to `[-10%, +4%]`.
3. Linearly map into staking band `[20%, 70%]`.
4. Apply 7-day annualized volatility overlay:
   - `0.80 < vol_7d ≤ 1.00` → set staking to **10%**.
   - `vol_7d > 1.00` → set staking to **0%**, i.e. 90% into PT stablecoins.
5. Apply **1-day execution lag**: `w_staking_signal = w_staking.shift(1)`.

Stable PT allocation is simply:

```python
w_liquid = 0.10
w_stable_PT_signal = 0.90 - w_staking_signal
w_total_alloc = w_liquid + w_staking_signal + w_stable_PT_signal
