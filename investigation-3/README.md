# Investigation 3: SK Hynix

Centres purely on SK Hynix (000660.KS). Investigation-2 covers US index funds
and broad assets; investigation-3 reuses the same scraping tactic but sets SK
Hynix as the correlation reference. Each investigation is self-contained.

## Scripts

`skhynix_correlation_ladder.py`
Scans a broad cross-asset universe (same set as the investigation-2 NASDAQ
scan) for daily Spearman and Pearson correlation against SK Hynix, then draws
one lollipop ladder of the strongest correlations. Two dollar measures (DXY
and USD/KRW) are pinned onto the ladder so the USD reading is always visible.
Outputs land in `data/`, `visuals/` and the headline PNG in `conclusion_3/`.

`krw_skhynix_correlation.py`
Focused two-asset pipeline: full-sample and 252-day rolling Pearson and
Spearman between USD/KRW and SK Hynix.

## USD against SK Hynix

The motivating question: semiconductors are priced in USD, so dollar strength
should register against SK Hynix even though the stock is quoted in KRW. The
scan separates the broad dollar from the won:

| Dollar measure | Symbol | Spearman | Pearson |
| --- | --- | --- | --- |
| Broad dollar index | DX-Y.NYB | −0.03 | −0.06 |
| Dollar ETF | UUP | −0.03 | −0.06 |
| Won pair | KRW=X | −0.11 | −0.14 |

Full sample, n = 3,937 (2015 to 2026). The broad dollar correlation is near
zero; DXY and UUP agree. The negative signal is roughly three times stronger
in the won pair than in the broad dollar. The "semis are USD-priced" intuition
does not show up in the broad dollar: SK Hynix's daily dollar sensitivity is
won-specific, not a general USD effect. The likely reason is that the two
channels offset for the broad dollar (dollar up lifts KRW-translated USD
revenue, but also signals risk-off), while against the won the risk-off,
EM-outflow channel dominates.
