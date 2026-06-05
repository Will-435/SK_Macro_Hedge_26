# Macro Correlation Scan

## 1. No Structural Hedge

Scanned 87 assets (incl. 22 crypto) vs NASDAQ. Strongest inverse: TLT, IEF, IEI, GOVT, SHY at ~−0.10 Spearman / −0.13 Pearson. Nothing clears the −0.5 threshold that I set (but this is arbitrary). No tradeable daily-frequency hedge exists.

## 2. Daily vs Long-Horizon

KRW vs SK Hynix daily log returns: **Pearson −0.14, Spearman −0.12** (n = 2,930). Rolling 252-day stays between −0.05 and −0.25.

KOSPI, SK Hynix and KRW have all trended together since 2021. The trend is anti-correlated. Daily returns do not see it. Two horizons, two models.

The SK Hynix Spearman ladder (`skhynix_spearman_ladder.png`) re-runs the cross-asset scan with SK Hynix as the reference instead of NASDAQ, and pins USD/KRW on regardless of rank. Full sample, n = 3,937: USD/KRW is the only negative rung at Spearman −0.11 / Pearson −0.14. Every strong rung is a positive co-mover — KOSPI +0.53, KOSDAQ +0.35, EWY +0.30, then EM / DM / semis ETFs. These are substitutes, not hedges. USD/KRW is the only structurally inverse asset against SK Hynix but the daily rank correlation is too weak to hedge on its own.

## 3. Driver is Yield Spreads

USD rolling correlations vs oil, Mag 7, QQQ all sit between −0.5 and +0.3. None dominate. 2022–24 strong-dollar regime tracks Fed hikes and Treasury yield spreads. Drop petrodollar / technodollar framings.

## 4. Gold = FX Leg

Gold vs USD: Spearman −0.42, Pearson −0.38. Stable across full sample and bearish NASDAQ days. Cleanest fact in the project. Use gold for the USD view, not for equity tail risk.

## 5. SK Hynix Stress-Day Gold: Watch, Don't Trade

74 days where SKH 4w ≤ P10 AND 21d RV ≥ P75. Gold averaged ~27 bps/day, hit rate 60%. Pearson 0.16, Spearman 0.06.

Both methods sit well below the 0.7 threshold. n = 74 is too small to size.

## 6. Three-Leg Framing

* **Equity.** Cut KR semi concentration. Do not rotate into US large-cap growth, crypto equities, or DM ETFs — all are NASDAQ substitutes (+0.6 to +0.99 Spearman).
* **FX.** Gold, sized to the bearish-USD view.
* **Equity tail.** Duration Treasuries (TLT, IEF). Weak (−0.10) but only available inverse.

Gold covers the FX leg. Do not double-count it as the tail leg.

## 7. Next Step: Forecast the KRW

Trade thesis now depends on the KRW path. Investigation 3 will forecast it directly.

**Feature set.** US Treasury yield spreads (2s10s, 3m10y), Fed funds path, KOSPI level + return, KRW realised vol, current-account proxy, gold–USD spot. Korean household debt and Korean public debt enter as separate series — different policy reaction functions.

**Options surface.** Layer USD/KRW options on top of the macro features. Pull three things from the surface:
* ATM implied vol → central tendency of the implied distribution.
* 25-delta risk reversal → directional skew.
* Term structure → forward evolution to 1Y.

The surface gives a market-implied distribution. Macro features alone give a point forecast.

**Models.** Run the same target through four families on identical train/val splits:
* GLM — linear baseline.
* GAM — smooth non-linear features (yield spreads, KRW RV).
* RF + isotonic regression — calibrated probabilistic output.
* NN — regime-conditional non-linearities the others can't capture.

**Promotion rule.** Any feature kept in the live model must clear Spearman ≥ 0.7 with the target.

## 8. Open Items Before Investigation 3

* Weekly + monthly inverse scans. Daily may under-count tradeable lower-frequency inverse correlations.
* Add gold and silver to the USD rolling-correlation chart.
* Bootstrap CI on the SKH bearish-regime gold result. n = 74 needs a confidence band.
* Move off yfinance for the options surface. Free-source USD/KRW options coverage is sparse; vendor feed (Bloomberg, Refinitiv) required.
