# Levered Dynamic Asset Allocation with Dual Momentum & Volatility Risk Control

This repository implements a **systematic ETH staking vault strategy** that combines:

- **Dual-horizon momentum–based asset allocation**
- **Trend-filtered dynamic leverage**
- **Volatility-based crash protection**
- **Stochastic DeFi yield and borrow modeling**

The objective is to **maximize long-term risk-adjusted returns while strictly controlling drawdowns, liquidation risk, and tail events**.

---

## 🔧 Core Strategy Architecture

The strategy is built from four independent but interacting engines:

---

### 1. Dynamic Leverage Engine (Trend-Based)

Leverage is applied to ETH staking based on multi-timeframe trend filters using:

- **EMA 8**
- **EMA 20**
- **EMA 50**

**Leverage Regimes:**

| Market Regime | Condition | Leverage |
|---------------|----------|----------|
| Strong Uptrend | Price > EMA50 & EMA8 > EMA20 | 2.0× |
| Mild Uptrend | Price > EMA50 | 1.5× |
| Early Trend | EMA8 > EMA20 | 1.0× |
| Bearish | Otherwise | 0× |

**Hard Stop-Loss Overlay:**
- If **Price < EMA50 → Max leverage capped at 0.5×**
- If **Price < 0.95 × EMA50 → Leverage forced to 0×**

This ensures leverage is only deployed during favorable market regimes and is aggressively reduced on trend failure.

---

### 2. Dual Momentum Asset Allocation Engine

Capital is split into:

- **10% Permanent Liquidity Reserve**
- **90% Dynamic Risk Capital**

The 90% risk bucket is allocated dynamically between:

- **ETH staking**
- **Short-duration stable PT tokens**

#### Step 1 – Structural Allocation (30-Day Momentum)
The **30-day ETH return** defines the long-term risk budget.  
Returns are clipped to a predefined range and mapped to a **base ETH staking weight between 20% and 70%**.

#### Step 2 – Tactical Acceleration (7-Day Momentum)
The **7-day ETH return** scales the base allocation:
- Positive short-term momentum increases exposure
- Weak momentum reduces exposure early

#### Step 3 – Capital Routing
The final ETH staking weight is applied to the 90% risk bucket.  
Any unused portion is allocated to **short-duration stable PT tokens** for capital preservation and yield stability.

---

### 3. Volatility-Based Crash Regime Filter

A dedicated crash-protection layer uses **7-day annualized realized ETH volatility**:

| Volatility Regime | Action |
|-------------------|--------|
| 80% – 100% | ETH staking capped at 10% |
| > 100% | ETH staking forced to 0% |

This layer acts as a **pure risk-off circuit breaker**, independent of trend or momentum, protecting against liquidation cascades and volatility shocks.

---

### 4. DeFi Yield & Leverage Modeling

- **ETH–stETH looping** is modeled using **stochastic staking and borrowing APYs** drawn from normal distributions to reflect realistic funding dynamics.
- **Stable PT tokens** are modeled as a **3× leveraged fixed carry strategy**.
- Net yield is computed as:

\[
\text{Net APY} =
L \cdot \text{Staking APY} - (L - 1)\cdot \text{Borrow APY}
\]

This separates **pure yield carry** from **price risk** and prevents the system from relying on unsustainable funding conditions.

---

## 📊 Key Risk Controls

- Dual momentum prevents late-cycle overexposure
- EMA regime filters prevent leverage in downtrends
- Volatility circuit breaker prevents crash liquidation
- Yield-leg decoupling prevents negative carry bleed
- Signal shifting ensures **no look-ahead bias**

---


