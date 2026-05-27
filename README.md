# SK_Macro_Hedge_26

South Korean tech equity fallback — macro options trade research.

This repository hosts the research pipeline behind a macro options trade that uses Gold as a fallback against concentrated exposure to South Korean semiconductor equities (Samsung Electronics, SK Hynix) and the global semiconductor ETF (SOXX).

## `Gold-time-series-analysis.py`

A self-contained pipeline that asks one question:

- When semiconductor equities suffer left-tail 4-week price moves, **and** when volatility regimes are elevated, how does gold behave?

The aim is to seperate the periods where gold actually offsets semi weakness from the periods where it co-moves with the stress.

## Volatility proxies

Historical per-stock implied volatility is not avaliable on free data sources (yfinance only returns current option snapshots, not a historic IV time series). The pipeline therefore uses two complementary proxies in parallel:

- **VIX** (Yahoo ticker `^VIX`) — the global equity-vol regime indicator.
- **21-day rolling realised volatility** of each semi ticker — a stock-specific IV proxy.

For every IV-dependant output the pipeline produces three variants, so that the result is not silently dependant on a single vol definition.

## The three regime variants

Each regime is a boolean mask over the daily calender. They are:

- **`no_vol_filter`** — left-quartile 4-week semi return only. No volatility condition. This is the "without VIX" output the brief asked for.
- **`high_vix`** — left-quartile 4-week semi return **and** VIX in its top quartile. This is the "with VIX" output.
- **`high_realised_vol`** — left-quartile 4-week semi return **and** that ticker's own 21-day annualised realised volatility in its top quartile.

The same three masks are then re-used everywhere downstream so that the rolling correlations, the gold summary stats and the correlation matrices are all directly comparable across regimes.

## What the script computes

- Daily log returns for gold, Samsung, SK Hynix and SOXX.
- 4-week (20 trading day) rolling log returns per semi.
- 21-day annualised realised volatility per semi.
- Trading volume per semi, smoothed with a 20-day moving average.
- Conditional gold summary statistics per (ticker, regime) — mean, median, std, hit-rate, plus Pearson and Spearman correlation with the semi on the regime days.
- Rolling 252-day conditional correlation between gold and each semi, restricted within each window to the regime days.
- Regime-filtered correlation matrices over `[gold, samsung, skhynix, soxx]`, computed with both Pearson and Spearman rank correlation.

## Outputs

All artefacts are written to `outputs/`.

### Plots

- `prices_and_vix.png` — normalised price panel with VIX on a secondary axis.
- `vix_regime.png` — VIX over time with the top-quartile regime highlighted.
- `realised_volatility.png` — 21-day annualised realised vol per semi.
- `four_week_return_distributions.png` — histogram of 4-week semi returns with the left-quartile threshhold marked.
- `trading_volume.png` — 20-day moving-average trading volume per semi.
- `conditional_gold_<regime>.png` — bar chart of gold mean return (bps) and gold hit-rate-positive per ticker, one file per regime.
- `rolling_corr_<regime>.png` — rolling 252-day conditional correlation between gold and each semi, one file per regime.
- `correlation_matrices_grid.png` — 3 by 2 grid of regime-filtered correlation matrices (rows = regime, columns = Pearson and Spearman).

### Tables and CSVs

- `prices.csv`, `volumes.csv`, `vix.csv` — raw aligned inputs.
- `log_returns.csv`, `four_week_returns.csv`, `realised_volatility.csv` — derived series.
- `rolling_corr_<ticker>_<regime>.csv` — the rolling correlation time series, one per ticker and regime.
- `corr_matrix_<method>_<regime>.csv` — the regime-filtered correlation matrices.
- `pipeline_summary.json` — full machine-readable run summary, including all conditional gold statistics and all six correlation matrices.

## How to read the output

- Look at the **conditional gold mean** for each regime. If it is meaningfully positive on left-tail semi days, gold is offsetting the stress on average.
- Compare the same number across regimes for one ticker. If gold's average return falls (or flips negative) once VIX is added to the filter, gold is failing to hedge in the precise regime where the trade actually needs it.
- The **rolling 252-day** plots show whether that hedging behaviour is stable over time or regime-dependant. A line that drifts upward through the sample is the canonical signature of a hedge that is decaying.
- The **correlation matrices** in `correlation_matrices_grid.png` give the snapshot view. Pearson captures linear co-movement; Spearman captures monotonic co-movement and is less sensitive to outliers, which matters alot on tail-filtered data.

## Sign convention and caveats

- Gold is the COMEX gold front-month future `GC=F`, quoted in USD per troy ounce.
- The pipeline is **correlational, not causal**. It describes the conditional dependance structure between gold and semi returns, which is the right object for a diversification or hedging argument, but does not identify the underlying mechanism.
- The high-vol thresholds use the full-sample 75th percentile. This is a fixed reference, not a rolling one, so the regime masks are stable across the run but tilted towards the tail of the realised distribution.
- Yahoo Finance is convenient but not authoritative; production runs should swap the data layer for a vendor feed (Bloomberg, Refinitiv) without changing the modelling code.
- VIX is a US-equity vol gauge. It is a reasonable proxy for global equity vol regimes but is not a Korean-specific stress indicator. The stock-specific realised vol regime is provided as a complementary view that does not have this geographic bias.

## Install and run

- `pip install yfinance pandas numpy scipy statsmodels matplotlib seaborn`
- `python Gold-time-series-analysis.py --start 2015-01-01 --end 2026-05-01`

CLI flags:

- `--start` — ISO date, default `2015-01-01`.
- `--end` — ISO date, default today (UTC).

The script logs progress to stderr and prints a compact end-of-run summary table; the full detail is in `outputs/pipeline_summary.json`.
