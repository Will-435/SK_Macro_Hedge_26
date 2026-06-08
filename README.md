# SK_Macro_Hedge_26

South Korean tech equity hedge research.

This repository holds the research behind a hedge for a concentrated long in SK
Hynix and Korean semiconductors. The work started as a direct gold-vs-semis
hedge thesis, broadened into a cross-asset diversification study, and now
centres on hedging the SK Hynix book directly. The current conclusion: the best
structural hedge is to reduce gross SK Hynix and semiconductor exposure and wait
for an AI/semiconductor catalyst before re-entering. No overlay tested improves
risk-adjusted return, so de-risking, not an offsetting short, is the action.

## Note

Verbose docstrings and inline explanations are left throughout as a record of
the workflow and reasoning.

## Repository layout

- `investigation-1/`: the original gold-vs-Korean-semis study and the SK Hynix
  liquidity deep dive.
- `investigation-2/`: cross-asset macro-correlation work on US assets: USD vs
  oil and AI proxies, and the inverse-rank scraper that ranks a broad universe
  against the NASDAQ Composite.
- `investigation-3/`: SK Hynix as the correlation reference: the same scrapers
  pointed at SK Hynix, plus focused two-asset pairs.
- `Master/`: the USD/KRW four-week return model (PCA plus neighbourhood
  bootstrap) and the report correlation-matrix renderer.
- `rep_metrics.cpp`: C++ engine that computes the hedge tear-sheet metrics from
  the directory data. Reads `rep_metrics_input.csv`, writes `rep_corr_matrix.csv`.
- `.env`: API keys, git-ignored and never committed.
- `requirements.txt`, `README.md`.

Each investigation and `Master/` is self-contained. Scripts write next to their
own `data/raw/<script>/`, `data/processed/<script>/` and `visuals/<script>/`, so
the pipelines regenerate independently and never collide. Parquet is the default
on-disk format; CSV is written only where another tool reads it. Each
investigation also carries a `conclusion_*/` directory holding the written
write-up and the curated figures.

## Investigation 1

Gold behaviour when semiconductor equities take left-tail four-week moves under
elevated volatility regimes; the aim is to separate the periods where gold
offsets semi weakness from those where it co-moves with the stress.

### Scripts

- `Gold-time-series-analysis.py`: Gold, Samsung, SK Hynix and SOXX. Uses VIX and
  per-stock realised volatility as the volatility proxies. Produces conditional
  gold statistics, rolling 252-day conditional correlations and regime-filtered
  correlation matrices under three regime variants.
- `deep_gold_SKH_liquidity_risk.py`: single-ticker SK Hynix deep dive using only
  the stock's own realised volatility. The regime requires the four-week return
  at or below its 10th percentile and the 21-day annualised realised volatility
  at or above its 75th percentile.

### Key takeaways

- Daily gold-vs-semi correlations are tiny in calm regimes and rise in stress,
  the canonical failure of a direct hedge.
- The SK Hynix bearish plus high-realised-vol sub-sample is the exception: gold
  averages roughly +36 basis points per day on the 74 qualifying days with a 61%
  hit rate. The sample is small and the regime rare, so it is suggestive, not
  tradeable on its own.

## Investigation 2

Cross-asset US work: once a direct hedge looked weak, the question became where
diversification actually comes from.

### Scripts

- `usd_oil_ai_rolling_correlation.py`: rolling 252-day and 63-day correlations
  between the US Dollar Index and each of WTI crude, an equal-weighted Mag 7
  basket and QQQ, plus correlation matrices over gold, USD and NASDAQ across the
  full sample and a bearish NASDAQ regime.
- `inverse_pearson_rank_scraper.py`: scans US sector, industry, broad-index and
  country ETFs, US Treasury yields, US and international bond ETFs, plus crypto
  tokens and crypto-exposed equities. Computes Pearson and Spearman correlations
  against the NASDAQ Composite and produces four ranked ladder diagrams.

### Key takeaways

- Gold vs USD holds a stable Spearman of roughly −0.4 across calm and bearish
  regimes, the cleanest single fact in the project and the basis for treating
  gold as an FX leg, not an equity hedge.
- Daily NASDAQ correlations max at roughly +0.99 (IWF, XLK) and roughly −0.10
  (long-duration Treasuries). No asset in the universe reaches −0.6.
- International sovereign yields are sourced from the FRED API (monthly OECD
  series), not yfinance, whose coverage of those yields is unreliable. Because
  they are monthly, they are correlated at monthly frequency rather than forced
  onto the daily grid.

## Investigation 3

SK Hynix as the correlation reference: the same scan tactic as investigation-2,
pointed at SK Hynix instead of the NASDAQ.

### Scripts

- `skhynix_correlation_ladder.py`: scans the broad cross-asset universe for
  daily Pearson and Spearman correlation against SK Hynix and draws a lollipop
  ladder of the strongest. The two dollar measures (DXY and USD/KRW) are pinned
  so the dollar reading is always visible.
- `krw_skhynix_correlation.py`: full-sample and 252-day rolling Pearson and
  Spearman between USD/KRW and SK Hynix.
- `ftse_skhynix_pearson_correlation.py`: Pearson and rolling Pearson between the
  FTSE 100 and SK Hynix, with the non-overlapping London/Seoul session caveat
  documented in the file.
- `skhynix_kospi_returns_overlay.py`: one-year indexed return paths of SK Hynix
  and KOSPI on a common base.

### Key takeaways

- SK Hynix co-moves strongly with KOSPI (Pearson roughly +0.65 full sample, +0.83
  over the trailing year), so KOSPI is the only candidate strong enough to hedge.
- The broad dollar (DXY) is near zero against SK Hynix; the inverse relationship
  lives in the won pair specifically (USD/KRW roughly −0.15 full, −0.25 recent),
  not in the dollar generally. Gold is near zero against the book.

## Master

The USD/KRW four-week (twenty trading day) return model.

### Scripts

- `pca_factors.py`: pulls factors from yfinance and the FRED API, applies a
  per-feature stationarity transform, standardises, and fits PCA. The Bank of
  Korea ECOS feed is not used because it is restricted to Korean residents; the
  macro series come from FRED instead.
- `bootstrap_pdf.py`: projects the current factor state into PCA space, finds
  its neighbourhood by Mahalanobis distance, and runs a stationary block
  bootstrap on the matched forward returns to produce a conditional PDF and fan
  chart.
- `main.py`: runs the two stages end to end and persists every artefact.
- `Master.py`: renders the cross-asset correlation matrix written by
  `rep_metrics.cpp` as a heatmap PNG for the report appendix. It loads the C++
  output and only draws it; no values are recomputed here.

## Hedge metrics engine

`rep_metrics.cpp` is the C++ engine behind the hedge tear sheet. It reads
`rep_metrics_input.csv`, a plain extraction of the directory parquet files
(SK Hynix, KOSPI, USD/KRW, DXY, gold, US semis, Mag 7 and SK Hynix volume), and
computes correlations, the SK Hynix to KOSPI beta, hedged-versus-unhedged
performance, four-week scenario analysis, a gradient-descent and gradient-ascent
optimisation of the hedge ratio, and the cross-asset correlation matrix. All
quantitative report figures originate here so the calculation lives in one place.
The window is the trailing 252 trading days, set by `BACKTEST_WINDOW`.

### Key takeaways

- Trailing 252-day SK Hynix to KOSPI beta 1.53 (1.39 full sample); roughly 69% of
  the book's variance is KOSPI beta.
- Shorting KOSPI lowers the Sharpe ratio (3.20 unhedged to 1.43 hedged): it sheds
  a profitable, positively-correlated exposure. The gradient ascent finds the
  Sharpe-maximising hedge ratio is negative (−0.56), so no short hedge raises
  risk-adjusted return. Reducing gross exposure is the only Sharpe-consistent
  action.

## Data sources and API keys

- yfinance supplies all daily price series.
- The FRED API supplies the macro series. The key is read from `.env` at the
  repository root as `FRED_API`, which is git-ignored, with a placeholder
  fallback so an un-keyed checkout still imports. Set your own key in `.env`
  before running anything that hits FRED.
- The pipelines are correlational, not causal. They describe conditional
  dependence structures, the right object for a hedging argument, but do not
  identify a mechanism.
- yfinance is convenient but not authoritative; a production run should swap the
  data layer for a vendor feed without changing the modelling code.

## Visual style

- Figures favour a red, green and blue palette. Per-ticker colour assignments are
  fixed across each pipeline for consistency.
- Global font sizes are controlled by `TITLE_FONT` and `TEXT_FONT` at the top of
  each script.
- Every figure carries a descriptive caption directly below the chart defining
  any symbol used.

## Install and run

- `pip install -r requirements.txt`. The pinned dependencies are `numpy`,
  `pandas`, `matplotlib`, `seaborn`, `yfinance`, `scipy` and `pyarrow`; `requests`
  arrives with `yfinance` and is used for the FRED calls.
- The C++ engine needs a C++17 compiler. Build and run from the repository root:
  - `g++ -O2 -std=c++17 -o rep_metrics rep_metrics.cpp && ./rep_metrics`
- Run each Python pipeline from inside its own directory, for example:
  - `cd investigation-1 && python Gold-time-series-analysis.py --start 2015-01-01 --end 2026-05-01`
  - `cd investigation-2 && python inverse_pearson_rank_scraper.py --start 2015-01-01 --end 2026-05-01`
  - `cd investigation-3 && python skhynix_correlation_ladder.py --start 2015-01-01 --end 2026-05-01`
  - `cd Master && python main.py`
  - `cd Master && python Master.py`

Every Python pipeline accepts `--start` and `--end` as ISO dates, default
`2015-01-01` to today, logs progress to stderr, and writes a run summary to
`data/processed/<script>/pipeline_summary.json` inside its directory.

## Sign conventions and caveats

- Gold is the COMEX front-month future `GC=F`, quoted in USD per troy ounce.
- USD/KRW is quoted KRW per USD, so a positive return is won depreciation.
- High-vol thresholds use the full-sample 75th percentile, a fixed reference, so
  the regime masks are stable but tilted to the tail.
- The trailing 252-day window in the SK Hynix data is an extreme melt-up, which
  inflates return and Sharpe figures. The correlation, beta and drawdown figures
  are robust to it; the absolute return and Sharpe should be read as
  regime-specific, not steady-state.
