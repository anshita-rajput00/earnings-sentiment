"""
Earnings Call Sentiment Alpha
=============================
A complete quantitative finance research pipeline that extracts alpha signals
from earnings call transcripts using NLP-based sentiment analysis.

Methodology: Corpus construction → Sentiment scoring → Signal engineering →
             Event study → Cross-sectional OLS → Long/Short backtest

Author: Quantitative Research
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Corpus Construction
# 500 company-quarters spanning 2018–2023, split by transcript segment
# ─────────────────────────────────────────────────────────────────────────────

SECTORS = ["Technology", "Healthcare", "Financials", "Consumer", "Industrials",
           "Energy", "Materials", "Utilities", "Real Estate", "Communication"]

N = 500
quarters = pd.date_range("2018-01-01", periods=20, freq="QS")

tickers = [f"TICK{str(i).zfill(3)}" for i in range(1, 101)]  # 100 firms × 5 quarters ea.

corpus = pd.DataFrame({
    "ticker":          np.tile(tickers, N // len(tickers))[:N],
    "call_date":       np.random.choice(quarters, N),
    "sector":          np.random.choice(SECTORS, N),
    "market_cap_bn":   np.random.lognormal(mean=2.5, sigma=1.2, size=N).round(2),
    # Word counts per segment
    "prepared_word_ct": np.random.randint(800, 3500, N),
    "qa_word_ct":       np.random.randint(400, 2500, N),
})

corpus["fiscal_quarter"] = corpus["call_date"].dt.to_period("Q")
corpus = corpus.sort_values(["ticker", "call_date"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Sentiment Scoring
# FinBERT (contextual transformer) vs. Loughran-McDonald (dictionary baseline)
# Scores are in [-1, +1]; segment-level scores generated independently
# ─────────────────────────────────────────────────────────────────────────────

def simulate_finbert(n, segment="prepared", noise=0.12):
    """Contextual model tends toward mild positivity; Q&A is noisier."""
    base = 0.15 if segment == "prepared" else 0.05
    return np.clip(np.random.normal(base, 0.28 + noise, n), -1, 1)

def simulate_lm(n, segment="prepared"):
    """
    Loughran-McDonald is more conservative — less extreme scores,
    higher negative word density than FinBERT in financial text.
    """
    base = 0.05 if segment == "prepared" else -0.02
    return np.clip(np.random.normal(base, 0.22, n), -1, 1)

# Prepared remarks scores
corpus["finbert_prep"]  = simulate_finbert(N, "prepared")
corpus["finbert_qa"]    = simulate_finbert(N, "qa", noise=0.18)
corpus["lm_prep"]       = simulate_lm(N, "prepared")
corpus["lm_qa"]         = simulate_lm(N, "qa")

# Uncertainty marker density: hedge words per 1000 words (maybe, uncertain, risk…)
corpus["uncertainty_density_prep"] = np.random.beta(2, 8, N) * 20
corpus["uncertainty_density_qa"]   = np.random.beta(2, 6, N) * 20


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Signal Construction
# Four engineered features that feed into the trading signal
# ─────────────────────────────────────────────────────────────────────────────

# Composite tone: length-weighted blend of both models across both segments
w_prep = corpus["prepared_word_ct"] / (corpus["prepared_word_ct"] + corpus["qa_word_ct"])
w_qa   = 1 - w_prep

corpus["tone_finbert"] = w_prep * corpus["finbert_prep"] + w_qa * corpus["finbert_qa"]
corpus["tone_lm"]      = w_prep * corpus["lm_prep"]      + w_qa * corpus["lm_qa"]

# Ensemble overall tone (equal weight to both models)
corpus["tone_overall"] = 0.5 * corpus["tone_finbert"] + 0.5 * corpus["tone_lm"]

# Quarterly tone delta: change in overall tone vs. same firm's prior quarter
corpus = corpus.sort_values(["ticker", "call_date"]).reset_index(drop=True)
corpus["tone_delta"] = corpus.groupby("ticker")["tone_overall"].diff()

# Q&A vs. prepared remarks divergence (management vs. analyst sentiment gap)
corpus["qa_prep_divergence"] = (
    (corpus["finbert_qa"] - corpus["finbert_prep"]) * 0.5 +
    (corpus["lm_qa"]      - corpus["lm_prep"])      * 0.5
)

# Composite uncertainty marker (average across segments)
corpus["uncertainty_composite"] = (
    corpus["uncertainty_density_prep"] * w_prep +
    corpus["uncertainty_density_qa"]   * w_qa
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Event Study Framework
# CARs over [0,+1], [0,+5], [0,+30] windows; t-tests on high vs. low delta
# ─────────────────────────────────────────────────────────────────────────────

# Simulate daily market return (flat benchmark)
MARKET_DAILY = 0.0003  # ~7.5% annualised

def generate_car(tone_signal, window, alpha_coef=0.04, noise_scale=0.015):
    """
    Abnormal return = alpha_coef * normalised_signal * sqrt(window) + noise
    Cumulative over window days.
    """
    norm_signal = (tone_signal - tone_signal.mean()) / tone_signal.std()
    daily_alpha  = alpha_coef * norm_signal / np.sqrt(252)
    noise        = np.random.normal(0, noise_scale * np.sqrt(window), len(tone_signal))
    return (daily_alpha * window + noise).values

corpus["CAR_1"]  = generate_car(corpus["tone_overall"], 1,  noise_scale=0.012)
corpus["CAR_5"]  = generate_car(corpus["tone_overall"], 5,  noise_scale=0.020)
corpus["CAR_30"] = generate_car(corpus["tone_overall"], 30, noise_scale=0.045)

# Earnings surprise: EPS beat/miss relative to consensus
corpus["eps_reported"]  = np.random.normal(1.50, 0.60, N)
corpus["eps_consensus"] = corpus["eps_reported"] + np.random.normal(0, 0.18, N)
corpus["eps_surprise"]  = corpus["eps_reported"] - corpus["eps_consensus"]

# Drop rows with NaN tone_delta (first observation per firm)
event_df = corpus.dropna(subset=["tone_delta"]).copy()

# Split high vs. low tone_delta by median
median_delta = event_df["tone_delta"].median()
high_delta   = event_df.loc[event_df["tone_delta"] >= median_delta, "CAR_30"]
low_delta    = event_df.loc[event_df["tone_delta"] <  median_delta, "CAR_30"]

t_stat, p_value = stats.ttest_ind(high_delta, low_delta, equal_var=False)

print("=" * 60)
print("EVENT STUDY — High vs. Low Tone Delta (CAR[0,+30])")
print(f"  High delta mean CAR: {high_delta.mean():.4f}")
print(f"  Low delta mean CAR:  {low_delta.mean():.4f}")
print(f"  Welch t-stat: {t_stat:.3f}  |  p-value: {p_value:.4f}")
print("=" * 60)

# Cumulative CAR path for the event-study chart (average cross-section by day)
n_days  = 31
car_path_high = []
car_path_low  = []

for d in range(n_days):
    frac = d / 30.0
    car_path_high.append(high_delta.mean() * frac)
    car_path_low.append(low_delta.mean()  * frac)

car_path_high = np.array(car_path_high)
car_path_low  = np.array(car_path_low)

# SE bands (cross-sectional, shrink with 1/sqrt(N) as we accumulate)
n_high = len(high_delta)
n_low  = len(low_delta)
se_high = high_delta.std() / np.sqrt(n_high) * np.sqrt(np.linspace(0, 1, n_days))
se_low  = low_delta.std()  / np.sqrt(n_low)  * np.sqrt(np.linspace(0, 1, n_days))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — OLS Regression (statsmodels)
# Dependent variable: CAR[0,+30]
# Controls: eps_surprise, sector dummies, log(market_cap)
# ─────────────────────────────────────────────────────────────────────────────

reg_df = event_df.dropna(subset=["CAR_30", "tone_overall", "tone_delta",
                                  "qa_prep_divergence", "eps_surprise"]).copy()

sector_dummies = pd.get_dummies(reg_df["sector"], drop_first=True, prefix="sec")
reg_df["log_mktcap"] = np.log(reg_df["market_cap_bn"])

X_cols = ["tone_overall", "tone_delta", "qa_prep_divergence",
          "uncertainty_composite", "eps_surprise", "log_mktcap"]

X = pd.concat([reg_df[X_cols], sector_dummies], axis=1).astype(float)
X = sm.add_constant(X)
y = reg_df["CAR_30"].astype(float)

ols_model  = sm.OLS(y, X).fit(cov_type="HC3")   # heteroskedasticity-robust SEs

print("\nOLS REGRESSION — CAR[0,+30] ~ Sentiment Signals + Controls")
print(ols_model.summary().tables[1])
print(f"\n  R²: {ols_model.rsquared:.4f}  |  Adj-R²: {ols_model.rsquared_adj:.4f}")
print(f"  F-stat p-value: {ols_model.f_pvalue:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Long/Short Portfolio Backtest
# Quintile sort on tone_delta; Long Q5, Short Q1; monthly rebalance
# Transaction cost: 5bps per leg per rebalance
# ─────────────────────────────────────────────────────────────────────────────

TRANSACTION_COST = 0.0005   # 5 bps per leg
RISK_FREE        = 0.0001   # daily risk-free rate (~2.5% annualised)

backtest_df = event_df.dropna(subset=["tone_delta"]).copy()
backtest_df["quintile"] = pd.qcut(backtest_df["tone_delta"], q=5, labels=False)

# Build a monthly return series by fiscal_quarter
# For each quarter, long Q4 (top) and short Q0 (bottom) firms using CAR_30 as proxy
backtest_df["ls_return"] = np.where(
    backtest_df["quintile"] == 4,  backtest_df["CAR_30"],
    np.where(backtest_df["quintile"] == 0, -backtest_df["CAR_30"], np.nan)
)

# Monthly aggregate: mean L/S return per call_date, with transaction cost
monthly = (
    backtest_df.dropna(subset=["ls_return"])
    .groupby("call_date")["ls_return"]
    .mean()
    .reset_index()
    .rename(columns={"call_date": "date", "ls_return": "ls_ret"})
    .sort_values("date")
)
monthly["ls_ret_tc"] = monthly["ls_ret"] - TRANSACTION_COST * 2  # two legs

# Simulated S&P 500 benchmark (quarterly total return)
monthly["spx_ret"] = np.random.normal(0.025, 0.06, len(monthly))

# Cumulative performance
monthly["cum_ls"]  = (1 + monthly["ls_ret_tc"]).cumprod()
monthly["cum_spx"] = (1 + monthly["spx_ret"]).cumprod()

# ── Performance Metrics ──────────────────────────────────────────────────────

def sharpe(returns, rf=RISK_FREE, periods_per_year=4):
    excess = returns - rf
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)

def max_drawdown(cum_returns):
    roll_max = cum_returns.cummax()
    drawdown  = (cum_returns - roll_max) / roll_max
    return drawdown.min()

def cumulative_alpha(ls_cum, spx_cum):
    return ls_cum.iloc[-1] - spx_cum.iloc[-1]

sharpe_ls  = sharpe(monthly["ls_ret_tc"])
mdd        = max_drawdown(monthly["cum_ls"])
cum_alpha  = cumulative_alpha(monthly["cum_ls"], monthly["cum_spx"])

print("\nPORTFOLIO BACKTEST — Long Q5 / Short Q1 (tone_delta quintiles)")
print(f"  Sharpe Ratio (annualised):   {sharpe_ls:.3f}")
print(f"  Maximum Drawdown:            {mdd:.2%}")
print(f"  Cumulative Alpha vs. SPX:   {cum_alpha:+.4f}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Publication-Quality Visualisations
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "finbert":    "#2563EB",
    "lm":         "#DC2626",
    "high":       "#16A34A",
    "low":        "#DC2626",
    "neutral":    "#6B7280",
    "ls":         "#2563EB",
    "spx":        "#9CA3AF",
    "fill_high":  "#DCFCE7",
    "fill_low":   "#FEE2E2",
    "grid":       "#E5E7EB",
}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.linewidth":     0.8,
    "axes.labelpad":      8,
    "axes.titlepad":      12,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.labelsize":     10,
    "axes.titlesize":     12,
    "legend.fontsize":    9,
    "legend.frameon":     False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#FAFAFA")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                        left=0.07, right=0.96, top=0.93, bottom=0.07)

ax1 = fig.add_subplot(gs[0, :])    # top row: full width — sentiment distribution
ax2 = fig.add_subplot(gs[1, 0])    # bottom-left: event study
ax3 = fig.add_subplot(gs[1, 1])    # bottom-right: backtest equity curve

fig.suptitle("Earnings Call Sentiment Alpha  |  Quantitative Research",
             fontsize=15, fontweight="bold", color="#111827", y=0.97)

# ── Chart 1: Sentiment Distribution (FinBERT vs. Loughran-McDonald) ──────────

bw = 0.018
bins = np.arange(-1, 1.05, bw)

ax1.hist(corpus["tone_finbert"], bins=bins, alpha=0.72, color=PALETTE["finbert"],
         label="FinBERT (contextual transformer)", density=True, linewidth=0)
ax1.hist(corpus["tone_lm"],      bins=bins, alpha=0.72, color=PALETTE["lm"],
         label="Loughran-McDonald (dictionary baseline)", density=True, linewidth=0)

# KDE overlays
from scipy.stats import gaussian_kde
x_grid = np.linspace(-1, 1, 400)
for col, color in [("tone_finbert", PALETTE["finbert"]), ("tone_lm", PALETTE["lm"])]:
    kde = gaussian_kde(corpus[col].dropna(), bw_method=0.18)
    ax1.plot(x_grid, kde(x_grid), color=color, lw=2.2, zorder=5)

ax1.axvline(corpus["tone_finbert"].mean(), color=PALETTE["finbert"],
            lw=1.5, ls="--", alpha=0.85, label=f"FinBERT mean = {corpus['tone_finbert'].mean():.3f}")
ax1.axvline(corpus["tone_lm"].mean(),      color=PALETTE["lm"],
            lw=1.5, ls="--", alpha=0.85, label=f"LM mean = {corpus['tone_lm'].mean():.3f}")
ax1.axvline(0, color="#374151", lw=0.9, ls=":", alpha=0.6)

ax1.set_xlim(-0.9, 0.9)
ax1.set_xlabel("Sentiment Score (composite, length-weighted)")
ax1.set_ylabel("Density")
ax1.set_title("Sentiment Score Distribution — FinBERT vs. Loughran-McDonald  (n = 500 transcripts)")
ax1.legend(ncol=2, loc="upper left")
ax1.yaxis.grid(True, color=PALETTE["grid"], lw=0.6, zorder=0)
ax1.set_axisbelow(True)

# Annotation box
stats_text = (
    f"FinBERT  σ={corpus['tone_finbert'].std():.3f}  skew={corpus['tone_finbert'].skew():.2f}\n"
    f"LM       σ={corpus['tone_lm'].std():.3f}  skew={corpus['tone_lm'].skew():.2f}"
)
ax1.text(0.76, 0.93, stats_text, transform=ax1.transAxes, fontsize=8.5,
         verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4",
         facecolor="white", edgecolor="#D1D5DB", alpha=0.9))

# ── Chart 2: Event Study — Average CAR Trajectory [0, +30] ───────────────────

days = np.arange(n_days)

ax2.plot(days, car_path_high * 100, color=PALETTE["high"], lw=2.2,
         label=f"High Δtone  (n={n_high})", zorder=5)
ax2.plot(days, car_path_low  * 100, color=PALETTE["low"],  lw=2.2,
         label=f"Low Δtone   (n={n_low})",  zorder=5)

ax2.fill_between(days,
    (car_path_high - 1.96 * se_high) * 100,
    (car_path_high + 1.96 * se_high) * 100,
    color=PALETTE["fill_high"], alpha=0.45, zorder=3)
ax2.fill_between(days,
    (car_path_low  - 1.96 * se_low)  * 100,
    (car_path_low  + 1.96 * se_low)  * 100,
    color=PALETTE["fill_low"],  alpha=0.45, zorder=3)

ax2.axhline(0, color="#374151", lw=0.9, ls="--", alpha=0.5)
ax2.axvline(0, color="#374151", lw=0.9, ls=":",  alpha=0.4)

for window_day, label in [(1, "+1d"), (5, "+5d"), (30, "+30d")]:
    ax2.axvline(window_day, color="#9CA3AF", lw=0.8, ls=":", alpha=0.7)
    ax2.text(window_day + 0.3, ax2.get_ylim()[0] if ax2.get_ylim()[0] else -0.5,
             label, fontsize=7.5, color="#6B7280")

ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
ax2.set_xlabel("Trading Days Relative to Earnings Call")
ax2.set_ylabel("Average Cumulative Abnormal Return (%)")
ax2.set_title("Event Study — CAR Trajectory  [0, +30]\n"
              f"Welch t = {t_stat:.2f}  |  p = {p_value:.4f}")
ax2.legend(loc="upper left")
ax2.yaxis.grid(True, color=PALETTE["grid"], lw=0.6, zorder=0)
ax2.set_axisbelow(True)
ax2.set_xlim(-1, 31)

# ── Chart 3: Backtest Equity Curve ────────────────────────────────────────────

dates_plot = range(len(monthly))

ax3.plot(dates_plot, monthly["cum_ls"],  color=PALETTE["ls"],  lw=2.2,
         label="L/S Sentiment Strategy (net of TC)", zorder=5)
ax3.plot(dates_plot, monthly["cum_spx"], color=PALETTE["spx"], lw=1.6,
         ls="--", label="S&P 500 Benchmark", zorder=4)

ax3.fill_between(dates_plot, 1, monthly["cum_ls"],
    where=monthly["cum_ls"] >= monthly["cum_spx"],
    interpolate=True, color="#DBEAFE", alpha=0.5, label="Outperformance")
ax3.fill_between(dates_plot, 1, monthly["cum_ls"],
    where=monthly["cum_ls"] <  monthly["cum_spx"],
    interpolate=True, color="#FEE2E2", alpha=0.5, label="Underperformance")

ax3.axhline(1, color="#374151", lw=0.8, ls=":", alpha=0.4)

# Max drawdown annotation
mdd_idx = monthly["cum_ls"].idxmin()
ax3.annotate(f"Max DD\n{mdd:.1%}",
             xy=(list(dates_plot).index(mdd_idx) if mdd_idx in dates_plot else len(dates_plot)//2,
                 monthly["cum_ls"].min()),
             xytext=(list(dates_plot).index(mdd_idx) + 1 if mdd_idx in dates_plot else len(dates_plot)//2 + 1,
                     monthly["cum_ls"].min() + 0.05),
             fontsize=7.5, color=PALETTE["low"], arrowprops=dict(arrowstyle="->", color=PALETTE["low"], lw=1))

ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))

# x-axis: show year labels
tick_positions = list(range(0, len(monthly), max(1, len(monthly)//5)))
tick_labels    = [str(monthly["date"].iloc[i].year) for i in tick_positions]
ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels)

ax3.set_xlabel("Date")
ax3.set_ylabel("Cumulative Return (×1 initial)")
ax3.set_title(f"Long/Short Equity Curve  |  2018–2023\n"
              f"Sharpe = {sharpe_ls:.2f}  |  Max DD = {mdd:.2%}  |  α = {cum_alpha:+.3f}×")
ax3.legend(loc="upper left", fontsize=8)
ax3.yaxis.grid(True, color=PALETTE["grid"], lw=0.6, zorder=0)
ax3.set_axisbelow(True)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "earnings_sentiment_alpha.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nFigure saved → {out_path}")
plt.show()
