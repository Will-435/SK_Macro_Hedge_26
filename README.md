# SK_Macro_Hedge_26

South Korean tech equity fallback — macro options trade research.

This repository hosts the research pipeline behind a macro options trade that
uses Gold as a fallback / hedge against concentrated exposure to South Korean
semiconductor equities (Samsung Electronics, SK Hynix) and the KRW.

---

## Contents

| File | Purpose |
| --- | --- |
| `Gold-time-series-analysis.py` | GAM-based time series analysis of Gold vs. South Korean semis and KRW. |

---

## `Gold-time-series-analysis.py`

A self-contained pipeline that quantifies the **diversification value** and
**hedging value** of Gold against South Korean semiconductor equity exposure,
conditioning on the KRW/USD regime. It is the statistical core of the broader
SK_Macro_Hedge_26 research effort; downstream modules (options structuring,
scenario PnL, sizing) consume its outputs.

### Why a GAM

The joint distribution of Gold returns, KRW moves, and Korean tech equity
returns is well known to be **non-linear and regime-dependent** — gold's hedging
power tends to strengthen in left-tail FX-stress episodes and weaken in benign
risk-on regimes. A linear factor regression collapses that structure into a
single slope. A Generalized Additive Model preserves it via smooth terms
without committing to a specific parametric form, and a tensor-product
interaction term lets gold's marginal effect on semi returns flex with the FX
state directly.

### Model

For each Korean semiconductor ticker `i`:

```
r_i,t = f1(r_gold,t) + f2(r_krw,t) + f3(vol_krw,t)
      + f4(r_i,t-1) + te(r_gold,t, r_krw,t) + eps_t
```

- `f1..f4` are penalised cubic-spline smooths.
- `te(·, ·)` is a tensor-product smooth capturing the Gold × KRW interaction
  that drives the conditional-hedging interpretation.
- `f4` on the lagged target absorbs short-run autocorrelation (AR(1)-like).

### Pipeline steps

1. **Data acquisition** — Yahoo Finance pull of Gold (`GC=F`), USDKRW
   (`KRW=X`), Samsung (`005930.KS`), SK Hynix (`000660.KS`), and `SOXX` as a
   global semiconductor control. Aligned to a common trading calendar.
2. **Returns + stationarity** — log returns, ADF and KPSS diagnostics per
   series.
3. **Descriptives** — annualised vol, skew, kurtosis.
4. **Correlation structure** — static Pearson plus 90-day rolling correlations
   between each semi and Gold (and KRW vs Gold as reference).
5. **GAM fit** — gridsearch over smoothing penalties; Wald p-values per term;
   pseudo-R², effective DoF, AIC.
6. **Hedge / diversification metrics**
   - OLS beta of semi returns on gold returns.
   - **Tail** correlations conditional on the bottom decile of semi returns
     (the regime that actually matters for a hedge).
   - Empirical lower-tail dependence coefficient.
   - Diversification ratio of a 50/50 semi+gold sleeve vs. semi-only.
7. **Diagnostics** — Ljung-Box on residuals (lag 10) and an expanding-window
   out-of-sample pseudo-R² backtest.
8. **Artefacts** — written to `outputs/`:
   - `prices.csv`, `log_returns.csv`, `descriptive_stats.csv`
   - `normalised_prices.png`, `rolling_correlations.png`
   - `pdp_<ticker>.png` — partial-dependence panels including the 2D `te`
     interaction contour
   - `pipeline_summary.json` — full machine-readable run summary

### Interpreting the output

- **`gold_p`, `interaction_p`** in the GAM summary: significance of the gold
  main effect and the gold×KRW interaction. The interaction is the key term for
  the hedging story — a strong `te` p-value indicates gold's effect on semi
  returns is materially different across KRW regimes.
- **`tail_corr_gold`** vs **`full_sample_corr_gold`**: a tail correlation
  meaningfully *below* the full-sample correlation is the empirical signature
  of useful left-tail hedging. If tail correlation rises in stress, gold is
  diversifying on average but not when you need it.
- **`tail_dependence_lower`**: probability of joint left-tail events. Lower is
  better for a hedge.
- **`diversification_ratio`**: > 1 indicates volatility reduction from
  blending; the further above 1, the larger the diversification benefit.

### Sign conventions / caveats

- `KRW=X` is quoted **KRW per USD**, so `r_krw > 0` means **KRW depreciation**.
  Read all KRW partial-dependence plots with that convention in mind.
- Daily returns are used; for a robustness check, re-run with weekly returns to
  dampen non-synchronous trading between Seoul (KRX) and COMEX gold.
- The analysis is **correlational, not causal**. It characterises the
  conditional dependence structure, which is the right object for a
  diversification / hedging claim, but does not identify mechanism.
- Yahoo Finance is convenient but not authoritative; production runs should
  switch the data layer to a vendor feed (Bloomberg, Refinitiv) without
  changing the modelling code.

### Install & run

```bash
pip install yfinance pandas numpy scipy statsmodels pygam matplotlib seaborn

python Gold-time-series-analysis.py --start 2015-01-01 --end 2026-05-01
```

CLI flags:

- `--start` — ISO date, default `2015-01-01`
- `--end`   — ISO date, default today (UTC)

The script logs progress to stderr and prints a compact end-of-run summary
table; richer detail is in `outputs/pipeline_summary.json`.
