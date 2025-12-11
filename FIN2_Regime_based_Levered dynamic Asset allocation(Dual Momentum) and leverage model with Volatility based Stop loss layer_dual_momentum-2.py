#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[3]:


df = pd.read_csv('/Users/samarveer/Downloads/ETH_BTC_price_2023_07_10_2025.csv')
df['Date']=pd.to_datetime(df['Date']).dt.date
df['ETH_Daily_chng']=df["ETH Price"].pct_change()
df['ETH_Daily_chng']=df['ETH_Daily_chng'].fillna(df['ETH_Daily_chng'].dropna().iloc[0])
# annualized realized vol: ETH 
df['ETH_ret_abs'] = df['ETH_Daily_chng'].abs()
df['ETH_vol_7d'] = df['ETH_Daily_chng'].rolling(7).std() * np.sqrt(365)   
df['ETH_vol_30d'] = df['ETH_Daily_chng'].rolling(30).std() * np.sqrt(365)   
df['ETH_vol_90d'] = df['ETH_Daily_chng'].rolling(90).std() * np.sqrt(365)   
# annualized realized vol: BTC
df['BTC_Daily_chng']=df["BTC Price"].pct_change()
df['BTC_Daily_chng']=df['BTC_Daily_chng'].fillna(df['BTC_Daily_chng'].dropna().iloc[0])
df['BTC_ret_abs']=df['BTC_Daily_chng'].abs()
df['BTC_vol_7d']= df['BTC_Daily_chng'].rolling(7).std() * np.sqrt(365)
df['BTC_vol_30d']= df['BTC_Daily_chng'].rolling(30).std() * np.sqrt(365)
df['BTC_vol_90d']= df['BTC_Daily_chng'].rolling(90).std() * np.sqrt(365)
# 30d momentum: ETH
df['mom_eth_30'] = df['ETH Price'] / df['ETH Price'].shift(30) - 1
# 30d momentum: BTC
df['mom_btc_30'] = df['BTC Price'] / df['BTC Price'].shift(30) - 1
# 180d drawdown from rolling ATH : ETH
eth_ath_180 = df['ETH Price'].rolling(180).max()
df['dd_eth_180'] = df['ETH Price'] / eth_ath_180 - 1
# 180d drawdown from rolling ATH : BTC
btc_ath_180 = df['BTC Price'].rolling(180).max()
df['dd_btc_180'] = df['BTC Price'] / btc_ath_180 - 1
# --- 1. ETH/BTC Price Ratio ---
df['rs_eth_btc_30_log'] = (
    np.log(df['ETH Price'] / df['BTC Price']) -
    np.log(df['ETH Price'].shift(30) / df['BTC Price'].shift(30))
)
df = df.dropna(subset=['rs_eth_btc_30_log'])

feature_cols= ['ETH_Daily_chng', 'BTC_Daily_chng',
    'ETH_vol_30d', 'BTC_vol_30d',
    'mom_eth_30', 'mom_btc_30',
    'dd_eth_180', 'dd_btc_180',
    'rs_eth_btc_30_log']
features= df[feature_cols].dropna().copy()

scaler = StandardScaler()
X = scaler.fit_transform(features)

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(
    n_components=4,
    covariance_type='full',
    random_state=42,
    n_init=10
)
regimes = gmm.fit_predict(X)

features['regime'] = regimes
df.loc[features.index, 'regime'] = regimes
regime_summary = df.groupby('regime').agg({
    'ETH_ret_abs': 'mean',
    'ETH_vol_30d': 'mean',
    'mom_eth_30': 'mean',
    'dd_eth_180': 'mean',
    'rs_eth_btc_30_log': 'mean'
})
# 1. Bull = highest momentum, shallowest drawdown
bull_regime = regime_summary['mom_eth_30'].idxmax()

# 2. Panic = deepest drawdown, highest vol
panic_regime = (
    regime_summary['dd_eth_180'].idxmin()
)

# Remaining two regimes → Accumulation vs Distribution
remaining = [r for r in regime_summary.index
             if r not in [bull_regime, panic_regime]]

# Accumulation: less damaged (higher dd, closer to 0)
accum_regime = regime_summary.loc[remaining]['dd_eth_180'].idxmax()

# Distribution: the remaining one
dist_regime = [r for r in remaining if r != accum_regime][0]

regime_map = {
    bull_regime:  'Bull',
    accum_regime: 'Accumulation',
    dist_regime:  'Distribution',
    panic_regime: 'Panic'
}

df['regime_name'] = df['regime'].map(regime_map)
regime_risk_weight = {
    'Bull':          1.0,   # max risk
    'Accumulation':  0.6,   # medium
    'Distribution':  0.3,   # low
    'Panic':         0.0    # off
}

df['regime_risk'] = df['regime_name'].map(regime_risk_weight)
df['regime_risk_live'] = df['regime_risk'].shift(1)
df['regime_name_live'] = df['regime_name'].shift(1)
df


## Final version of Levered Dynamic asset allocation model


LT=0.83
curr_stk_apy=0.0289
curr_bor_apy=0.0018

mu_stk_apy= 0.0007
std_stk_apy= 0.0059
mu_brw_apy= 0.0054
std_brw_apy= 0.0161
INITIAL_CAPITAL=1000
Fixed_Lev=2
rf_annual=0.035
ANNUAL_DAYS=365
R_MIN = -0.10   # -10%
R_MAX =  0.05   # +5%
W_MIN =  0.40   # 40% staking at worst
W_MAX =  0.70   # 70% max staking
Eth_return_duration_long=30
Eth_return_duration_short=7
stble_pt_apy=0.06
stble_pt_brw_apy=0.04
stable_lev=3




# Monte Carlo sims for stk_apy and brw_apy
T_days=len(df)
N_sims=10000
seed     = 42
sim_idx = 0
rng      = np.random.default_rng(seed)
stk_apy_sims= rng.normal(mu_stk_apy, std_stk_apy, size=(T_days, N_sims)) 
brw_apy_sims= rng.normal(mu_brw_apy, std_brw_apy, size=(T_days, N_sims)) 
stk_apy_sims= (curr_stk_apy*(1+stk_apy_sims))
brw_apy_sims= (curr_bor_apy*(1+brw_apy_sims))
df['stk_apy'] = stk_apy_sims[:, sim_idx] 
df['brw_apy'] = brw_apy_sims[:, sim_idx]



df['EMA_8']=df['ETH Price'].ewm(span=8, adjust=False).mean()  
df['EMA_20']=df['ETH Price'].ewm(span=20, adjust=False).mean()  
df['EMA_50']=df['ETH Price'].ewm(span=50, adjust=False).mean()
df['Trend']=df['EMA_8']>df['EMA_20']
df['Major trend']=df['ETH Price']>df['EMA_50']

df['eth_ret_30d'] = df['ETH Price'] / df['ETH Price'].shift(Eth_return_duration_long) - 1
r_long = df['eth_ret_30d']
df['eth_ret_7d'] = df['ETH Price'] / df['ETH Price'].shift(Eth_return_duration_short) - 1
r_short = df['eth_ret_7d']




# Clip R_7d into [-10%, +4%] range
r_clipped = r_long.clip(lower=R_MIN, upper=R_MAX)

# Linear mapping:
# w_staking = W_MIN + ((R - R_MIN) / (R_MAX - R_MIN)) * (W_MAX - W_MIN)
df['w_staking'] = W_MIN + ((r_clipped - R_MIN) / (R_MAX - R_MIN)) * (W_MAX - W_MIN)

# Safety clamp (should already be in [0.2, 0.7], but explicit is nice)
df['w_staking'] = df['w_staking'].clip(lower=W_MIN, upper=W_MAX)

short_neg = -0.03
short_pos = 0.03
r_short_clip = r_short.clip(lower=short_neg, upper=short_pos)
# Map to [0.5 , 1.2] risk multiplier
df['m_short'] = 0.5 + ((r_short_clip - short_neg) / (short_pos - short_neg)) * (1.2 - 0.5)
df['m_short'] = df['m_short'].clip(lower=0.5, upper=1.2)

df['w_staking_raw'] = df['w_staking'] * df['m_short']
df['w_staking_raw'] = df['w_staking_raw'].clip(lower=0, upper=W_MAX)

df.loc[(df['ETH_vol_7d'] > 0.8) & (df['ETH_vol_7d'] <= 1.0), 'w_staking_raw'] = 0.10     # severe panic
df.loc[df['ETH_vol_7d'] > 1, 'w_staking_raw'] = 0     # high volatility
df['w_staking_signal'] = df['w_staking_raw'].shift(1).fillna(W_MIN)
## Final asset allocation weights

df['w_staking_signal']= df['w_staking_signal']*df['regime_risk_live']
# === Step 3 – Liquid and stable-PT weights ===

# Always 10% liquid
df['w_liquid'] = 0.10


# Just in case: no negative weights
df['w_stable_PT_signal'] = (0.90 - df['w_staking_signal']).clip(lower=0.0)
# Optional: check total allocation (should be ~1.0)
df['w_total_alloc'] = df['w_staking_signal'] + df['w_stable_PT_signal'] + df['w_liquid']

# === Step 4 – Asset-level returns for allocation model ===

# Unlevered Staking (LSTs) ≈ ETH exposure + staking yield
df['ret_staking_unlev'] = df['ETH_Daily_chng'] + df['stk_apy'] / ANNUAL_DAYS

# Unlevered Short-duration stable-PT (USDe, sGHO) – assume rf_annual for now
df['ret_stable_PT_unlev'] = stble_pt_apy / ANNUAL_DAYS

# Liquid assets – assume 0% return (pure dry powder)
df['ret_liquid_asset'] = 0.0

# === Step 5 – Unlevered Portfolio daily return from allocation ===
df['ret_alloc_unlev'] = (
    df['w_staking_signal']    * df['ret_staking_unlev'] +
    df['w_stable_PT_signal']  * df['ret_stable_PT_unlev'] +
    df['w_liquid']     * df['ret_liquid_asset'])





def get_dyn_lev(row):
    price = row['ETH Price']
    ema8 = row['EMA_8']
    ema20 = row['EMA_20']
    ema50 = row['EMA_50']

    # Strong uptrend → 2x
    if (price > ema50) and (ema8 > ema20):
        lev= 2

    # Mild uptrend → 1.5x
    elif price > ema50:
        lev= 1.5

    # Early trend → 1x
    elif ema8 > ema20:
        lev= 1

    # Bearish → 0x
    else:
        lev= 0
    
    ## stop loss layer
    if price<ema50*0.95:
        return 0
    if price<ema50:
        lev = min(lev,0.5)
    return lev


# Apply the function row-by-row
df['dyn_lev'] = df.apply(get_dyn_lev, axis=1)

# Optional: safety clamp
df['dyn_lev'] = df['dyn_lev'].clip(lower=0, upper=2)
df['dyn_lev_signal'] = df['dyn_lev'].shift(7).fillna(1.0)
df['dyn_lev_signal']= df['regime_risk_live']*df['dyn_lev_signal']


## Final staking 

# Leveraged Allocation model returns
# Levered Staking return
ret_staking_lev_raw = (
    df['dyn_lev_signal'] * df['ETH_Daily_chng']
    + (df['dyn_lev_signal'] * df['stk_apy'] - (df['dyn_lev_signal'] - 1.0) * df['brw_apy']) / ANNUAL_DAYS
)
df['ret_staking_lev'] = np.where(
    df['dyn_lev_signal'] > 1.0,
    ret_staking_lev_raw,
    df['ret_staking_unlev']  # pure staking leg if no leverage
)

# Levered PT stable return
df['ret_stable_PT_lev'] = (stable_lev * stble_pt_apy - ((stable_lev-1)* stble_pt_brw_apy)) / 365



df['ret_alloc_lev'] = (
    df['w_staking_signal']    * df['ret_staking_lev'] +
    df['w_stable_PT_signal']  * df['ret_stable_PT_lev'] +
    df['w_liquid']     * df['ret_liquid_asset']
)


collateral_val = df['dyn_lev_signal'] * INITIAL_CAPITAL * (1 + df['ETH_Daily_chng'])
debt_val = (df['dyn_lev_signal'] - 1.0) * INITIAL_CAPITAL
df['HF'] = np.where(
    debt_val <= 0,
    np.inf,
    (collateral_val * LT) / debt_val
)
df['liq_status'] = np.where(np.isfinite(df['HF']) & (df['HF'] < 1.0), 'Yes', 'No')
collateral_val_fix = Fixed_Lev * INITIAL_CAPITAL * (1 + df['ETH_Daily_chng'])
debt_val_fix = (Fixed_Lev - 1.0) * INITIAL_CAPITAL
df['HF_fix'] = np.where(
    debt_val_fix <= 0,
    np.inf,
    (collateral_val_fix * LT) / debt_val_fix
)    
df['liq_status_fix'] = np.where(np.isfinite(df['HF_fix']) & (df['HF_fix'] < 1.0), 'Yes', 'No')

df["Net_APY_Model1"] = (df['dyn_lev_signal'] * df['stk_apy'] - (df['dyn_lev_signal'] - 1) * df['brw_apy']) / 365
df['Fix_APY_Model']= (Fixed_Lev * df['stk_apy'] - ((Fixed_Lev-1)* df['brw_apy'])) / 365
df['ret_bh'] = df['ETH_Daily_chng'] + df['stk_apy']/365
# Fixed Leverage (with APY added)
df['ret_fixed'] = df['ETH_Daily_chng'] * Fixed_Lev + df['Fix_APY_Model']
# Dynamic Leverage (with APY)
df['ret_dyn'] = df['ETH_Daily_chng'] * df['dyn_lev_signal'] + df['Net_APY_Model1']
def returns_to_value(rets, initial=1000):
    val = (1 + rets).cumprod() * initial
    val.iloc[0] = initial
    return val
df['val_bh'] = returns_to_value(df['ret_bh'], INITIAL_CAPITAL)
rets_fixed_clamped = df['ret_fixed'].clip(lower=-0.999999)
df['val_fixed'] = returns_to_value(rets_fixed_clamped, INITIAL_CAPITAL)
df['val_dyn_preliq'] = returns_to_value(df['ret_dyn'], INITIAL_CAPITAL)
df['val_alloc_lev']= returns_to_value(df['ret_alloc_lev'], INITIAL_CAPITAL)
df['val_fixed_before']=df['val_fixed'].shift(1)
df['val_fixed_before']=df['val_fixed_before'].fillna(INITIAL_CAPITAL)
df['val_dyn_preliq_before']=df['val_dyn_preliq'].shift(1)
df['val_dyn_preliq_before']=df['val_dyn_preliq_before'].fillna(INITIAL_CAPITAL)
df['val_alloc_before']=df['val_alloc_lev'].shift(1)
df['val_alloc_before']=df['val_alloc_before'].fillna(INITIAL_CAPITAL)

## Pre Collateral, Debt and HF:Fixed
df['collateral_before_Fixed'] = Fixed_Lev * df['val_fixed_before']
df['collateral_after_Fixed'] = df['collateral_before_Fixed'] * (1 + df['ETH_Daily_chng'])
df['debt_before_Fixed'] = (Fixed_Lev - 1.0) * df['val_fixed_before']
df['HF_fixed_before']= (df['collateral_after_Fixed'] * LT) / df['debt_before_Fixed']
df['liq_status_fix_pre'] = np.where(np.isfinite(df['HF_fixed_before']) & (df['HF_fixed_before'] < 1.0), 'Yes', 'No')## Pre Collateral, Debt and HF:Dynamic

## Pre Collateral, Debt and HF:Dynamic
lev    = df['dyn_lev_signal']
equity = df['val_dyn_preliq_before']
chg    = df['ETH_Daily_chng']

# 1) Collateral before
# If lev == 0 → you're out of the loop, just holding equity
df['collateral_before_dyn'] = np.where(lev > 0, lev * equity, equity)

# 2) Debt before
# Only have debt when lev > 1
df['debt_before_dyn'] = np.where(lev > 1.0, (lev - 1.0) * equity, 0.0)

# 3) Collateral after price move
df['collateral_after_dyn'] = df['collateral_before_dyn'] * (1 + chg)

# 4) HF: handle zero-debt rows cleanly
df['HF_dyn_before'] = np.where(
    df['debt_before_dyn'] > 0,
    (df['collateral_after_dyn'] * LT) / df['debt_before_dyn'],
    np.inf  # no debt → no liquidation risk
)

# 5) Liquidation status
df['liq_status_dyn_pre'] = np.where(
    (df['debt_before_dyn'] > 0) & np.isfinite(df['HF_dyn_before']) & (df['HF_dyn_before'] < 1.0),
    'Yes',
    'No'
)

## Pre Collateral, Debt and HF:Dynamic Asset allocation model
equity1=df['val_alloc_before']
w_stk_sig = df['w_staking_signal']
equity_stk = w_stk_sig * equity1
## Asset allocation pre collateral
df['collateral_before_ass']= np.where(lev > 1.0, lev * equity_stk, equity_stk)
## Asset allocation Pre Debt
df['debt_before_ass']= np.where(lev > 1.0, (lev - 1.0) * equity_stk, 0.0)
##Collateral after price move: Dynamic Asset allocation model
df['collateral_after_ass']= df['collateral_before_ass']*(1+chg)
##HF: Dynamic Asset allocation model
df['HF_ass_before'] = np.where(
    df['debt_before_ass'] > 0,
    (df['collateral_after_ass'] * LT) / df['debt_before_ass'],
    np.inf
)# no debt → no liquidation risk
##Liquidation status: Dynamic Asset allocation model
df['liq_status_ass_pre'] = np.where(
    (df['debt_before_ass'] > 0) & np.isfinite(df['HF_ass_before']) & (df['HF_ass_before'] < 1.0),
    'Yes',
    'No'
)


# Sharpe Ratio
def sharpe(daily_ret, rf_annual=0.035):
    # Convert annual risk-free rate to daily
    rf_daily = (1 + rf_annual)**(1/365) - 1

    dr = daily_ret.fillna(0)

    # Excess daily returns over risk-free
    excess = dr - rf_daily

    if excess.std() == 0:
        return np.nan
    
    # Annualized Sharpe
    return (excess.mean() / excess.std()) * np.sqrt(365)

df['ret_bh_port'] = df['val_bh'].pct_change()
df['ret_fixed_port'] = df['val_fixed'].pct_change()
df['ret_dyn_port'] = df['val_dyn_preliq'].pct_change()
df['ret_ass_port']= df['val_alloc_lev'].pct_change()

sharpe_bh = sharpe(df['ret_bh_port'])
sharpe_fixed = sharpe(df['ret_fixed_port'])
sharpe_dyn = sharpe(df['ret_dyn_port'])
sharpe_ass= sharpe(df['ret_ass_port'])

def max_drawdown(values):
    running_max = values.cummax()
    dd = (values - running_max) / running_max
    max_dd = -dd.min()       # positive
    return max_dd
max_dd_bh = max_drawdown(df['val_bh'])
max_dd_fixed = max_drawdown(df['val_fixed'])
max_dd_dyn = max_drawdown(df['val_dyn_preliq'])
max_dd_ass= max_drawdown(df['val_alloc_lev'])

def CAGR(values):
    values = values.dropna()
    start = values.iloc[0]
    end = values.iloc[-1]
    n_days = len(values)
    
    if start <= 0 or n_days < 2:
        return np.nan
    
    years = n_days / 365
    return (end / start)**(1/years) - 1
cagr_bh  = CAGR(df['val_bh'])
cagr_fixed = CAGR(df['val_fixed'])
cagr_dyn = CAGR(df['val_dyn_preliq'])
cagr_ass = CAGR(df['val_alloc_lev'])

def sortino(daily_vals, rf_annual=0, mar_annual=0.0):
    rets = daily_vals.pct_change().dropna()

    # convert annual to daily rates
    rf_daily = (1 + rf_annual)**(1/365) - 1
    mar_daily = (1 + mar_annual)**(1/365) - 1
    
    # excess returns
    excess = rets - rf_daily
    
    # downside deviation: returns below MAR
    downside = excess[excess < mar_daily]
    
    if downside.std(ddof=0) == 0:
        return np.nan
    
    sortino_daily = excess.mean() / downside.std()
    
    # annualize
    return sortino_daily * np.sqrt(365)
sortino_bh=sortino(df['val_bh'])
sortino_fixed=sortino(df['val_fixed'])
sortino_dyn=sortino(df['val_dyn_preliq'])
sortino_ass=sortino(df['val_alloc_lev'])

def calmar(values):
    cagr = CAGR(values)
    mdd = max_drawdown(values)
    
    if mdd == 0:
        return np.nan
    
    return cagr / mdd
calmar_bh=calmar(df['val_bh'])
calmar_fixed=calmar(df['val_fixed'])
calmar_dyn=calmar(df['val_dyn_preliq'])
calmar_ass=calmar(df['val_alloc_lev'])

df['stk_apy']   # annual %
df['brw_apy']   # annual %
df['dyn_lev_signal'] # daily leverage signal
df['net_apy_daily'] = (df['dyn_lev_signal'] * df['stk_apy'] - (df['dyn_lev_signal'] - 1) * df['brw_apy']) / 365

df['cum_net_apy'] = (1+df['net_apy_daily']).cumprod()
initial_capital = 1000
df['val_from_net_apy'] = initial_capital * df['cum_net_apy']


print("         Performance Metrics(01/2023 - 10/2025)       ")
print("---------------------------------------------------")
print(" SHARPE RATIOS")
print("---------------------------------------------------")
print(f"Buy & Hold:                                                {sharpe_bh:.4f}")
print(f"Fixed 2x leverage:                                         {sharpe_fixed:.4f}")
#print(f"Dynamic Model:                             {sharpe_dyn:.4f}")
print(f"Regime based: Levered Asset allocation Dynamic Model:      {sharpe_ass:.4f}")

print("---------------------------------------------------")
print(" SORTINO RATIOS")
print("---------------------------------------------------")
print(f"Buy & Hold:                                                {sortino_bh:.4f}")
print(f"Fixed 2x leverage:                                         {sortino_fixed:.4f}")
#print(f"Dynamic LTV:                                 {sortino_dyn:.4f}")
print(f"Regime based: Levered Asset allocation Dynamic Model:      {sortino_ass:.4f}")

print("---------------------------------------------------")
print(" Drawdown")
print("---------------------------------------------------")
print(f"Buy & Hold:                                                {max_dd_bh:.4f}")
print(f"Fixed 2x leverage:                                         {max_dd_fixed:.4f}")
#print(f"Dynamic LTV:                                 {max_dd_dyn:.4f}")
print(f"Regime based: Levered Asset allocation Dynamic Model:      {max_dd_ass:.4f}")

print("---------------------------------------------------")
print(" CALMAR RATIOS")
print("---------------------------------------------------")
print(f"Buy & Hold:                                                {calmar_bh:.4f}")
print(f"Fixed 2x leverage:                                         {calmar_fixed:.4f}")
#print(f"Dynamic LTV:                                 {calmar_dyn:.4f}")
print(f"Regime based: Levered Asset allocation Dynamic Model:      {calmar_ass:.4f}")

print("---------------------------------------------------")
print(" CAGR")
print("---------------------------------------------------")
print(f"Buy & Hold:                                                {cagr_bh:.4f}")
print(f"Fixed 2x leverage:                                         {cagr_fixed:.4f}")
#print(f"Dynamic LTV:                                 {cagr_dyn:.4f}")
print(f"Regime based: Levered Asset allocation Dynamic Model:      {cagr_ass:.4f}")


#print("\n---------------------------------------------------")
#print(" FINAL VALUES")
#print("---------------------------------------------------")
#print(f"Buy & Hold:                               ${df['val_bh'].iloc[-1]:.2f}")
#print(f"Fixed 2x:                                 ${df['val_fixed'].iloc[-1]:.2f}")
#print(f"Dynamic LTV:                              ${df['val_dyn_preliq'].iloc[-1]:.2f}")
#print(f"Levered Asset allocation Dynamic Model:   ${df['val_alloc_lev'].iloc[-1]:.2f}")

print("\n---------------------------------------------------")
print(" FINAL ROI")
print("---------------------------------------------------")
print(f"Buy & Hold:                                                {(df['val_bh'].iloc[-1]-1000)/1000:.2f}")
print(f"Fixed 2x leverage:                                         {(df['val_fixed'].iloc[-1]-1000)/1000:.2f}")
#print(f"Dynamic LTV:                              {(df['val_dyn_preliq'].iloc[-1]-1000)/1000:.2f}")
print(f"Regime based: Levered Asset allocation Dynamic Model:      {(df['val_alloc_lev'].iloc[-1]-1000)/1000:.2f}")




def print_apy_stats(df):
    
    print("\n-------------------------------------------------\n")
    print("------------ APY Simulation Summary(01/2023 - 10/2025) ------------")
    
    print("\nStaking APY (simulated)")
    print(f"Mean:   {df['stk_apy'].mean():.4%}")
    print(f"Median: {df['stk_apy'].median():.4%}")
    print(f"Std:    {df['stk_apy'].std():.4%}")
    print(f"Min:    {df['stk_apy'].min():.4%}")
    print(f"Max:    {df['stk_apy'].max():.4%}")

    print("\nBorrow APY (simulated)")
    print(f"Mean:   {df['brw_apy'].mean():.4%}")
    print(f"Median: {df['brw_apy'].median():.4%}")
    print(f"Std:    {df['brw_apy'].std():.4%}")
    print(f"Min:    {df['brw_apy'].min():.4%}")
    print(f"Max:    {df['brw_apy'].max():.4%}")
    

    print(f"Cumulative NET APY:    {df['cum_net_apy'].max():.4%}")
    
    print("\n-------------------------------------------------\n")
    print("------------ Levered Stable APY Simulation Summary ------------")
    
    
    print(f"Cumulative NET APY:    {(df['ret_stable_PT_lev']*365).max():.4%}")


    print("\n-------------------------------------------------\n")

print_apy_stats(df)

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['val_bh'], label="Buy & Hold", linewidth=2)
plt.plot(df['Date'], df['val_fixed'], label="Fixed 2x", linewidth=2)
plt.plot(df['Date'], df['val_dyn_preliq'], label="Dynamic LTV Model", linewidth=2)
plt.plot(df['Date'], df['val_alloc_lev'], label="Levered Asset allocation Dynamic Model", linewidth=2)
plt.grid(alpha=0.3)
plt.legend()
plt.title("Strategy Comparison — Cumulative Portfolio Value(01/2023 - 10/2025)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['val_from_net_apy'], label="Cumulative Net APY Growth")
plt.grid(alpha=0.3)
plt.legend()
plt.title("Net APY Compounding Curve(01/2023 - 10/2025)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.show()


# In[30]:


liq_days = df[df['liq_status_ass_pre'] == 'Yes'][['Date', 'ETH_Daily_chng', 'HF_fixed_before']]
liq_days


# In[48]:


def backtest_alloc_lev(
    df,
    R_MIN=-0.10,          # long-mom lower bound
    R_MAX=0.04,           # long-mom upper bound
    W_MIN=0.20,           # min staking weight
    W_MAX=0.70,           # max staking weight
    initial_capital=1000,
    rf_annual=0.035,
    Eth_return_duration_long=30,
    Eth_return_duration_short=7,
    short_neg=-0.03,
    short_pos=0.03,
    stble_pt_apy=0.06,
    stble_pt_brw_apy=0.04,
    stable_lev=3,
):
    """
    Backtest the dynamic asset allocation + leverage model
    using the current regime, momentum, vol, and leverage logic.
    Assumes df already has:
        - ETH Price, ETH_Daily_chng
        - ETH_vol_7d
        - dyn_lev_signal      (already regime-scaled)
        - stk_apy, brw_apy    (annual, per day sims)
        - regime_risk_live    (0–1 regime multiplier)
    """

    tmp = df.copy()

    # === 1) Long & short ETH momentum ===
    tmp['eth_ret_long'] = tmp['ETH Price'] / tmp['ETH Price'].shift(Eth_return_duration_long) - 1
    tmp['eth_ret_short'] = tmp['ETH Price'] / tmp['ETH Price'].shift(Eth_return_duration_short) - 1

    r_long = tmp['eth_ret_long']
    r_short = tmp['eth_ret_short']

    # === 2) Long-momentum → base staking weight [W_MIN, W_MAX] ===
    r_clipped = r_long.clip(lower=R_MIN, upper=R_MAX)

    tmp['w_staking'] = W_MIN + ((r_clipped - R_MIN) / (R_MAX - R_MIN)) * (W_MAX - W_MIN)
    tmp['w_staking'] = tmp['w_staking'].clip(lower=W_MIN, upper=W_MAX)

    # === 3) Short-momentum risk multiplier m_short ∈ [0.5, 1.2] ===
    r_short_clip = r_short.clip(lower=short_neg, upper=short_pos)
    tmp['m_short'] = 0.5 + ((r_short_clip - short_neg) / (short_pos - short_neg)) * (1.2 - 0.5)
    tmp['m_short'] = tmp['m_short'].clip(lower=0.5, upper=1.2)

    # Combine long & short layers
    tmp['w_staking_raw'] = tmp['w_staking'] * tmp['m_short']
    tmp['w_staking_raw'] = tmp['w_staking_raw'].clip(lower=0.0, upper=W_MAX)

    # === 4) Vol-based emergency clamp (ETH_vol_7d) ===
    # assumes ETH_vol_7d is already annualized realized vol
    tmp.loc[(tmp['ETH_vol_7d'] > 0.8) & (tmp['ETH_vol_7d'] <= 1.0), 'w_staking_raw'] = 0.10
    tmp.loc[tmp['ETH_vol_7d'] > 1.0, 'w_staking_raw'] = 0.0

    # === 5) Execution lag + regime throttle ===
    # 1-day lag for execution, then scale by regime risk
    tmp['w_staking_signal'] = tmp['w_staking_raw'].shift(1).fillna(W_MIN)

    # if regime_risk_live missing, assume 1.0
    if 'regime_risk_live' in tmp.columns:
        tmp['w_staking_signal'] = tmp['w_staking_signal'] * tmp['regime_risk_live']
    else:
        tmp['regime_risk_live'] = 1.0  # just in case

    # === 6) Liquid & stable-PT weights ===
    tmp['w_liquid_signal'] = 0.10
    tmp['w_stable_PT_signal'] = (0.90 - tmp['w_staking_signal']).clip(lower=0.0)
    tmp['w_total_alloc'] = (
        tmp['w_staking_signal'] +
        tmp['w_stable_PT_signal'] +
        tmp['w_liquid_signal']
    )

    # === 7) Asset-level returns (same as main code) ===
    ANNUAL_DAYS = 365

    # Unlevered staking (ETH price + staking APY)
    tmp['ret_staking_unlev'] = tmp['ETH_Daily_chng'] + tmp['stk_apy'] / ANNUAL_DAYS

    # Dynamic leverage signal (already regime-scaled outside)
    L = tmp['dyn_lev_signal']

    # Levered staking return (only when L > 1)
    ret_staking_lev_raw = (
        L * tmp['ETH_Daily_chng']
        + (L * tmp['stk_apy'] - (L - 1.0) * tmp['brw_apy']) / ANNUAL_DAYS
    )

    # If no leverage (L <= 1) → fallback to unlevered staking leg
    tmp['ret_staking_lev'] = np.where(L > 1.0, ret_staking_lev_raw, tmp['ret_staking_unlev'])

    # Levered PT stable leg (same daily constant as your main code)
    tmp['ret_stable_PT_lev'] = (
        (stable_lev * stble_pt_apy - ((stable_lev - 1) * stble_pt_brw_apy)) / ANNUAL_DAYS
    )

    # Liquid leg (dry powder)
    tmp['ret_liquid_asset'] = 0.0

    # === 8) Portfolio daily return (allocation + leverage) ===
    tmp['ret_alloc_lev'] = (
        tmp['w_staking_signal']    * tmp['ret_staking_lev'] +
        tmp['w_stable_PT_signal']  * tmp['ret_stable_PT_lev'] +
        tmp['w_liquid_signal']     * tmp['ret_liquid_asset']
    )

    # === 9) Convert returns → value curve ===
    val = returns_to_value(tmp['ret_alloc_lev'], initial_capital)
    tmp['val_alloc_lev'] = val

    # === 10) Metrics: Sharpe, Max DD, CAGR, Calmar ===
    ret_port = val.pct_change()

    sh  = sharpe(ret_port, rf_annual=rf_annual)
    mdd = max_drawdown(val)
    cg  = CAGR(val)
    cm  = calmar(val)

    return {
        'R_MIN': R_MIN,
        'R_MAX': R_MAX,
        'W_MIN': W_MIN,
        'W_MAX': W_MAX,
        'Sharpe': sh,
        'MaxDD': mdd,
        'CAGR': cg,
        'Calmar': cm,
    }


# In[49]:


import itertools

R_MIN_grid = np.linspace(-0.60, -0.10, 6)   # -60, -50, -40, -30, -20, -10
R_MAX_grid = np.linspace(0.01, 0.05, 5)     # 1%, 2%, 3%, 4%, 5%
W_MIN_grid = np.linspace(0.20, 0.50, 4)     # 20%, 30%, 40%, 50%
W_MAX_grid = np.linspace(0.70, 0.95, 6)     # 70%, 75%, 80%, 85%, 90%, 95%

results = []

for R_MIN, R_MAX, W_MIN, W_MAX in itertools.product(R_MIN_grid, R_MAX_grid, W_MIN_grid, W_MAX_grid):
    # Enforce basic sanity: W_MAX > W_MIN, R_MAX > R_MIN
    if R_MAX <= R_MIN:
        continue
    if W_MAX <= W_MIN:
        continue

    metrics = backtest_alloc_lev(
        df,
        R_MIN=R_MIN,
        R_MAX=R_MAX,
        W_MIN=W_MIN,
        W_MAX=W_MAX,
        initial_capital=INITIAL_CAPITAL,
        rf_annual=rf_annual
    )
    results.append(metrics)

results_df = pd.DataFrame(results)


# In[52]:


best_sharpe_row = results_df.loc[results_df['Sharpe'].idxmax()]
print("Best by Sharpe:")
print(best_sharpe_row)


# In[51]:


filtered = results_df[results_df['MaxDD'] < 0.25]
best_filtered = filtered.loc[filtered['Sharpe'].idxmax()]
print("Best Sharpe with MaxDD < 60%:")
print(best_filtered)


# In[ ]:




