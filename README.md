# Earnings Call Sentiment Alpha

NLP-based alpha signal extraction from earnings call transcripts.
Full pipeline: sentiment scoring → event study → OLS regression → long/short backtest.
Runs in Google Colab, no API keys or external data needed.

---

## Why I Built This

Wanted to go beyond EPS beats as a signal. Earnings calls are dense with
forward-looking language that the market doesn't fully price in immediately.
Especially the Q&A segment, where management has less control over the narrative.
Built this to see if tone shifts are actually predictable and tradeable.

---

## Methodology

### 1. Corpus Construction
500 company-quarter observations, 100 firms across 20 quarters (2018–2023), 10 GICS sectors.
Each transcript split into Prepared Remarks and Q&A Session — treated as separate
signals, not pooled.

### 2. Sentiment Scoring

| Model | Type | Notes |
|---|---|---|
| FinBERT | Contextual transformer | Mild positive bias; noisier in Q&A |
| Loughran-McDonald | Finance-domain dictionary | More conservative, slight negative drift |

Both run in parallel. Scores are length-weighted by segment word count.

### 3. Signal Construction

- Composite Tone — length-weighted ensemble across both models and both segments
- Tone Delta — quarter-over-quarter change, same firm
- Q&A / Prepared Divergence — how much the analyst session deviates from the script
- Uncertainty Composite — hedge-word density weighted across segments

### 4. Event Study
CAR windows: [0,+1], [0,+5], [0,+30]. Welch t-test on high vs. low tone-delta cohorts.

Result: t = 3.62, p = 0.0003 — high tone-delta firms outperform over the following 30 days.

### 5. OLS Regression
HC3 robust standard errors. Controls for EPS surprise, log market cap, sector fixed effects.

CAR[0,+30] ~ tone_overall + tone_delta + qa_prep_divergence
           + uncertainty_composite + eps_surprise + log(market_cap) + sector_FE

- tone_overall: +0.618 (p < 0.001)
- qa_prep_divergence: +0.095 (p = 0.018)
- Adj R2: 0.123

### 6. Long/Short Backtest
Quintile sort on tone_delta. Long Q5, short Q1. Rebalanced quarterly. 5bps per leg transaction cost.

| Metric | Value |
|---|---|
| Sharpe (annualised) | 2.32 |
| Max Drawdown | -4.60% |
| Cumulative Alpha vs. SPX | +7.38x |

Note: these numbers come from statistically realistic simulated data — not live market returns.
The simulation does not account for earnings announcement clustering, sector momentum, or
liquidity constraints. Real-world performance would differ.

---

## Output

Single 300 DPI figure with three charts:

1. FinBERT vs. LM score distributions (histogram + KDE)
2. CAR trajectory [0,+30] with 95% confidence bands
3. L/S equity curve vs. SPX benchmark

---

## Run It

Colab:
!python earnings_call_sentiment_alpha.py

Local:
pip install numpy pandas statsmodels scipy matplotlib seaborn
python earnings_call_sentiment_alpha.py

Python 3.8+

---

## Stack

pandas, numpy, statsmodels, scipy, matplotlib, seaborn

---

## Known Limitations

- Simulated data — sentiment scores are statistically calibrated but not real FinBERT inference
- No survivorship bias correction in firm sample
- Backtest does not model borrow costs on the short leg
- Single-factor sort; real implementations would use a composite rank

---

## Author

[Anshita Rajput] — github.com/anshita-rajput00 — linkedin.com/in/anshitarajput








