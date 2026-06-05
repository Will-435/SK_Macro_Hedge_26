# The steps following the conclusions from investigation 1 and 2

## Key takeaways

* There is no structual hedge against SK Hynix and south korean semis
* No strong inverse (<-0.5) Pearson and Spearman rank correlation scores for scraped yfinance tickers
* This could be misleading - KOSPI and KS.000660 have had mostly + gradient since '21
  * This coincides with a sharp fall in KRW/USD forex spot price since '21
  * Investigation 3 will be to begin investigating the Spearman rank and pearson correlation between SK Hynix, KOSPI and the KRW
  * In the form of correlation matricies AND a 252 day rolling correlation line chart
* The data shows taht the effects of a technodollar or petrodollar aren't major factors. Neither has a strong correlation - polarity is driven by US Treasury yield spreads
* The hit rate for a positive gold return over any given 4 weeks when SK Hynix realised volatility was in teh 75th percentile, and 4 week return was in the bottom 10th percentile, was ~60% with a mean return of ~27bps across all those days. (Pearson = 0.16, Spearman rank = 0.06)

## Next Steps 

* Prediction for the KRX Forecast 
* Do KOSPI and SK Hynix have strong person and Spearman rank correlations
* GLM, GAM, RF + Isotonic regression, NN - Which will best describe and predict the KRW and KOSPI
* Trying to carry a correlation of one macro event over from one asset to another is a bad idea. 
  * Before promoting a trade, I should be conservative with pearman corrolation scores (>= 0.7) - Any less is not a valid hedge
* In whatever model used, seperate household and public debt

## Further reading

* Could Crypto challenge the USD as a global reserve/replace weak currencies? - Iwan. 
* How much dos foreign exchange trading impact the value of currencies vs index funds and yield spreads?
* What if we use bootstrapping to generate synthetic data, in order to calculat ethe probability of what did happen, and then relate that to a GAM in order to update probabilities or create a sample of "if... then" conditions that lead us to use different GAM models trained on different possible historical paths from our bootstrap method? 