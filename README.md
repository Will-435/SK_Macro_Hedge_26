# SK_Macro_Hedge_26

South Korean tech equity fallback — macro trade hedge research.

This repository hosts the research pipeline behind a macro trade that started life as a direct gold-vs-Korean-semis hedge thesis. We have since broadened into a portfolio-diversification reframing, following a lack of a constant positive Pearson (or other) correlation. The work is now organised into two seperate investigations, each living in its own subdirectory at the top of the repo.

## Note:
Excessive doc strings and exlanations have been left throughout as a demonstration of my workflow.

## Repository layout

The top-level tree is intentionally simple:

- `investigation-1/` — the original gold-vs-Korean-semis study and the SK Hynix deep dive.
- `investigation-2/` — the broader macro-correlation work: USD vs oil and AI proxies, plus the inverse-Spearman scraper that ranks a broad universe of ETFs, bond yields and crypto assets against the NASDAQ Composite.
- `README.md` — this file.
- `requirements.txt` — Python dependancies pinned for the whole project.
- `venv/` — local virtual environment (not committed in production).

Each investigation directory is self-contained. The Python scripts that produce its outputs live next to their own `data/` and `visuals/` directories. Running a script always lands its raw downloads, processed tables and figures inside the same investigation directory it was launched from.

Inside each investigation:

- `data/raw/<script_name>/` — raw yfinance pulls cached as Parquet, one sub-directory per script.
- `data/processed/<script_name>/` — derived Parquet tables, CSVs where the user requested them, and the JSON run summary.
- `visuals/<script_name>/` — every figure as a PNG, one sub-directory per script.

This way the two investigations can be regenerated independantly and their outputs never collide.

## Investigation 1

Lives in `investigation-1/`. The original research question:

- When semiconductor equities suffer left-tail 4-week price moves, **and** when volatility regimes are elevated, how does gold behave?

The aim is to seperate the periods where gold actually offsets semi weakness from the periods where it co-moves with the stress.

### Scripts

- `Gold-time-series-analysis.py` — multi-ticker pipeline covering Gold, Samsung, SK Hynix and SOXX. Uses both VIX and per-stock realised volatility as IV proxies. Produces conditional gold statistics, rolling 252-day conditional correlations and regime-filtered correlation matrices under three regime variants (no vol filter, high VIX, high realised vol).
- `deep_gold_SKH_liquidity_risk.py` — single-ticker deep dive on SK Hynix that deliberately drops VIX and uses only the stock's own realised volatility. The headline regime requires the SKH 4-week return at or below its 10th percentile AND the 21-day annualised realised volatility at or above its 75th percentile.
- `conclusions 1/` — the written write-up of the investigation's findings.

### Key takeaways

- Daily-frequency gold-vs-Korean-semi correlations are tiny in calm regimes and rise in stress, which is the canonical "all correlations go to 1" failure of a direct hedge.
- The SK Hynix bearish + high-realised-vol sub-sample is the one exception: gold averages roughly **+36 basis points per day** on the 74 days that satisfy that filter, with a 61% hit rate. The sample is small and the regime is rare, so the result is suggestive rather than tradeable on its own.
- The rolling conditional correlation has drifted downwards through 2025 and turned negative in late 2025; whether this is a regime change or sample-size noise needs to be revisited as more regime days accumulate.

## Investigation 2

Lives in `investigation-2/`. The diversification reframing: once a direct hedge looked weak, the question became "what does the broader macro neighbourhood actually look like, and where is the diversification really coming from?"

### Scripts

- `usd_oil_ai_rolling_correlation.py` — rolling 252-day and 63-day correlations between the US Dollar Index and each of WTI crude oil, an equal-weighted Mag 7 basket and QQQ. Also produces a 2 by 2 grid of correlation matrices over gold, USD and NASDAQ comparing the full sample to a bearish NASDAQ regime.
- `inverse_pearson_rank_scraper.py` — scans a wide universe of US sector ETFs, US industry ETFs, broad index ETFs, country equity ETFs, US Treasury yields, US and international bond ETFs, plus eleven major crypto tokens and eleven crypto-exposed equities. Computes both Pearson and Spearman correlations against the NASDAQ Composite and produces four ranked ladder diagrams: top positive Spearman, top negative Spearman, top positive Pearson, top negative Pearson.

### Key takeaways

- Gold vs USD shows a stable Spearman correlation of roughly −0.4 across both calm and bearish-NASDAQ regimes. This is the cleanest single empirical fact in the whole project and is the empirical anchor for treating gold as the FX-leg of the diversification reframing rather than as an equity hedge.
- Daily-frequency NASDAQ correlations max out at roughly +0.99 (IWF, XLK) on the positive side and roughly −0.10 (long-duration Treasuries) on the negative side. No asset in a 87-strong universe reaches −0.6, including crypto.
- Crypto equities behave like high-beta NASDAQ proxies (Spearman around +0.5 to +0.7), not like the underlying tokens. Crypto tokens themselves cluster around +0.18 to +0.28 — uncorrelated-to-mildly-positive, not inverse.

## Visual style

- Every figure uses a deliberate red, green and blue palette throughout. Per-ticker colour assignments are fixed across each pipeline for visual consistancy.
- Global font sizes in each script are controlled by the two constants `TITLE_FONT` and `TEXT_FONT` at the top of the file. Editing either resizes every plot consistently.
- Every figure carries a descriptive caption directly below the chart that defines any symbol used in the figure (for example `rho`, `bps`, `r_X`, `Q1`, `RV_t`, `VIX`).

## Sign convention and caveats

- Gold is the COMEX gold front-month future `GC=F`, quoted in USD per troy ounce.
- The pipelines are **correlational, not causal**. They describe conditional dependance structures, which is the right object for a diversification or hedging argument, but does not identify the underlying mechanism.
- High-vol thresholds use the full-sample 75th percentile. This is a fixed reference, not a rolling one, so the regime masks are stable across the run but tilted towards the tail of the realised distribution.
- Yahoo Finance is convenient but not authoritative; production runs should swap the data layer for a vendor feed (Bloomberg, Refinitiv) without changing the modelling code.
- VIX is a US-equity vol gauge. It is a reasonable proxy for global equity vol regimes but is not a Korean-specific stress indicator. The stock-specific realised vol regime is provided as a complementary view that does not have this geographic bias.

## Install and run

- `pip install -r requirements.txt`
- From inside the investigation directory you want to run, invoke the script directly:
  - `cd investigation-1 && python Gold-time-series-analysis.py --start 2015-01-01 --end 2026-05-01`
  - `cd investigation-1 && python deep_gold_SKH_liquidity_risk.py --start 2015-01-01 --end 2026-05-01`
  - `cd investigation-2 && python usd_oil_ai_rolling_correlation.py --start 2015-01-01 --end 2026-05-01`
  - `cd investigation-2 && python inverse_pearson_rank_scraper.py --start 2015-01-01 --end 2026-05-01`

The dependancies pinned in `requirements.txt` are `numpy`, `pandas`, `matplotlib`, `seaborn`, `yfinance`, `scipy` and `pyarrow`. `pyarrow` is required for the Parquet output.

CLI flags accepted by every script:

- `--start` — ISO date, default `2015-01-01`.
- `--end` — ISO date, default today (UTC).

Each script logs progress to stderr and prints a compact end-of-run summary table; the full detail is in `data/processed/<script_name>/pipeline_summary.json` inside the relevant investigation directory.
