# Regime-Based Levered Asset Allocation Model (ETH Vault)

## Overview

This repository implements a **Regime-Based Levered Asset Allocation Engine for ETH**.
The core innovation of this system is a **data-driven market regime detector (Bull / Accumulation / Distribution / Panic)** that dynamically controls:

* ETH staking exposure
* Leverage (LTV)
* Stable/PT allocation
* Risk-off capital protection

Unlike traditional momentum or leverage-only strategies, this model is **state-aware**. It does **not attempt to predict price**. Instead, it **infers the current market regime** using multivariate ETH–BTC features and **adapts risk in real time**.

The result is:

* ✅ Significantly **lower drawdowns**
* ✅ Much higher **Sharpe, Sortino, and Calmar ratios**
* ✅ Survivability during crash regimes

---

## Key Innovation: Regime Engine

The regime model classifies **every trading day into one of four hidden market states**:

| Regime       | Meaning                      | Risk Stance          |
| ------------ | ---------------------------- | -------------------- |
| Bull         | Strong expansion             | Maximum risk allowed |
| Accumulation | Post-crash or base building  | Medium risk          |
| Distribution | Late-cycle, topping behavior | Low risk             |
| Panic        | Crash / systemic stress      | Risk fully off       |

This is achieved using a **Gaussian Mixture Model (GMM)** on a feature set derived from both **ETH and BTC**.

### Regime Features

The following features are computed daily:

* ETH daily return
* BTC daily return
* 30D realized volatility (ETH & BTC)
* 30D momentum (ETH & BTC)
* 180D drawdown from rolling ATH (ETH & BTC)
* 30D ETH/BTC relative strength (log)

These features capture:

* Market **direction**
* Market **stress (volatility)**
* Market **damage (drawdown)**
* **Relative leadership** between ETH and BTC

All features are standardized using `StandardScaler` before training the GMM.

---

## Why BTC is Included (Even if We Trade Only ETH)

BTC acts as the **systemic risk anchor of crypto liquidity**.
Even when investing only in ETH, BTC volatility and momentum provide **early warning signals** for:

* Liquidity contractions
* Distribution phases
* Crash onsets

Including BTC allows the regime engine to become **anticipatory rather than reactive**. This is a critical component of the drawdown reduction achieved by the system.

---

## Regime-Based Risk Control

Each detected regime maps to a **global risk multiplier**:

```text
Bull           → 1.0  (Full risk)
Accumulation   → 0.6
Distribution   → 0.3
Panic          → 0.0  (Risk fully off)
```

This multiplier throttles:

* ETH staking weight
* Dynamic leverage

For example:

* In **Bull**, the model allows full leverage and full ETH staking exposure.
* In **Accumulation**, only partial capital remains in ETH.
* In **Distribution**, the system aggressively de-risks into PT stables.
* In **Panic**, all directional ETH risk is shut off automatically.

The regime signal is **lagged by one day** to avoid look-ahead bias.

---

## Strategy Architecture

The strategy stack is organized into five layers:

1. **Regime Detection Layer**
   Multivariate GMM on ETH–BTC features

2. **Momentum & Volatility Layer**

   * 30D long momentum
   * 7D short momentum adjustment
   * 7D volatility emergency clamp

3. **Allocation Layer**

   * ETH staking sleeve
   * Stable PT yield sleeve
   * Liquid dry powder

4. **Dynamic Leverage Layer**
   EMA-based leverage with hard stop-loss logic

5. **Regime Risk Throttle**
   Overrides all signals during crash regimes

---

## Asset Exposure

The vault dynamically rotates between:

* **ETH Staking (Directional Risk + Yield)**
* **Levered ETH Exposure (when allowed by regime & trend)**
* **Stable PT Yield (Risk-Off Capital Parking)**
* **Liquid Capital (Dry Powder)**

---

## Backtest Performance Summary (Example Window)

Comparison across four strategies:

* Buy & Hold
* Fixed 2× Leverage
* Dynamic LTV Only
* **Regime-Based Levered Asset Allocation (This Model)**

Key outcomes observed:

* ✅ Highest Sharpe Ratio
* ✅ Highest Sortino Ratio
* ✅ Highest Calmar Ratio
* ✅ Lowest Maximum Drawdown
* ✅ Institutional-grade risk-adjusted performance

This confirms that the **regime engine is the dominant driver of drawdown suppression and stability**.

---

## Key Strengths

* Pure **state inference**, not prediction
* Fully **data-driven regime classification**
* Strong **tail-risk protection**
* Multi-layer risk control (momentum + vol + regime)
* Designed for **leveraged DeFi vaults**

---

## Intended Use Cases

* ETH leveraged yield vaults
* Delta-neutral ETH strategies with regime-aware throttles
* Risk-managed staking aggregation
* Institutional-grade composable DeFi vaults

---

## Research Status

This is a **quant research framework**, not financial advice.

The model is actively being:

* Stress tested across multiple market regimes
* Extended to multi-asset vaults
* Optimized for live deployment constraints

---

## Versioning

* **v1:** Dynamic LTV Only (archived)
* **v2 (Current):** Regime-Based Levered Asset Allocation Engine

---

## Author

Developed as part of a continuous **quant research & vault engineering project** focused on:

* Crypto regime modeling
* Capital preservation under leverage
* Retail-accessible institutional strategies

---

## Disclaimer

This repository is for **research and educational purposes only**.
It does not constitute financial advice. All trading involves significant risk, especially when leverage and DeFi protocols are involved.
