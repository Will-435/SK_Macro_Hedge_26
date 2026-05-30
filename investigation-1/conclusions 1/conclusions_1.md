# Regime Analysis: Gold vs. Semiconductors

## 1. Volatility Overlays
The VIX tracks broad U.S. S&P 500 variance and cannot proxy South Korean idiosyncratic risk. The 21-day realized volatility (RV) of the specific underlying asset is the required signal to identify localized stress. This is the best option withot access to IV.

## 2. Return Distributions
Rolling 4-week market returns are leptokurtic (fat-tailed), so unfortunately not Gaussian. this shows how volatile the returns can be. Our future risk models must account for this, maybe by using some other distribution for the mean returns - a t distribution maybe?

## 3. Regime Definition Thresholds
A 25th percentile return filter captures orderly market rotations, yielding a low-conviction (~55%) hit rate. To isolate true systemic liquidation events, regime constraints must be tightened to the 10th percentile for returns as per the DOI 10.1016/j.jfineco.2006.09.007 refrence (although they describe it as fairly arbitrary) and the 75th percentile for realized volatility. (75th can be reviewed later on as well)

## 4. Asset Selection
* **SK Hynix:** The optimal signal. It is a pure HBM proxy levered to global AI CapEx.
* **Samsung:** A diluted signal. The semiconductor beta is dragged down by samsungs other divisions which stretc across industries.
* **SOXX:**  Tracks U.S. domestic tech beta, not the South Korean export cycle. We'll drop this going forward.

## 5. KOSPI & FX Dynamics
It is functionally a memory chip duopoly, with Samsung and SK Hynix accounting for ~47% of the index's total market cap. A severe sell-off in these two assets crashes the KOSPI, triggering foreign capital repatriation and violent depreciation of the South Korean Won (KRW). In a tail event, Gold priced in KRW (XAUKRW) provides a direct hedge against this sovereign currency collapse. But this is only useful for a macroeconomic convexity trade which we will avoid. KOSPI is too overly exposed (~53%) to other Korean industries like electronics, financials and automotives. Exploring the relationship between gold and the KOSPI will lead to ill informed executions.