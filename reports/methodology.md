# Methods and interpretation

## Signal and one-pass processing
The primary factor is EMA40 of minus the trailing5-session compounded adjusted return times the positive part of today's volume relative to the previous5-session mean minus 1. The volume reference excludes today. EMA alpha is 2/41, initialized at the first valid input, emitted after 40 consecutive valid inputs and reset on missing observations. Span40 is not a hard40-day truncation.

Current-day average cross-sectional ranks enter a **single** joint OLS with historical industry dummies and log float cap. Residuals use population standard deviation. Extra controls enter this same joint regression; residuals are neither reranked nor neutralized again. Industry demeaning plus residualized regressors implements the full-dummy OLS through FWL.

The two targeted controls are EMA40(-5D return), and EMA40(tau / mean(prior20 tau)-1), tau=volume/circulating shares. Abnormal turnover is an attention / activity proxy, not a direct observation of optimism or panic. These two controls share source information with the factor. No residual-volatility control is included in the latest primary comparison.

## Execution and accounting
Signal t close -> execution t+1 close -> first holding return t+1 to t+2. Existing holdings earn the session return before rebalancing. Positive and negative signals are normalized separately to target+100% and-100% of NAV (gross target200%). Returns are relative to NAV, not divided again by gross notional.

Nontradable holdings freeze. Their remaining budget is deducted from each leg. Failed target orders are not allocated to other stocks using next-day availability. Cost is solved by fixed point on post-cost NAV and charged on actual bilateral traded amount.10bp means0.1% per executed amount, not a fee on gross exposure every day. Average one-way turnover is daily bilateral turnover/2, averaged over all sessions. Targets can differ from achieved exposures because of freezes/failed executions.

The main book is continuous across 2025; period curves are rebased for display. CAGR uses252 sessions/year; SR uses sample SD, rf=0; MDD includes initial NAV1. Borrow and financing costs are not included. Gross means before transaction, borrow, financing and impact costs.

## IC and Rank IC
At each signal date, correlate the final score with adjusted-close return from t+1 to t+1+h for h=1,5,21. IC is Pearson; Rank IC is Spearman with average ties. Entry and exit prices must be observed; endpoints cannot cross the named interval boundary. Five/21-session labels roll daily and overlap, rather than sampling weekly/month-end dates.

ICIR in this release is **unannualized mean/std**. Positive fraction is the fraction of valid daily correlations above zero. HAC Bartlett lag=max(21,h) estimates mean-correlation uncertainty; normal-approximation two-sided p-values and confidence intervals are descriptive. Main PNL mean tests use HAC63. These tests and local family adjustments do not correct the entire historical search. Cumulative IC is a correlation sum, not a portfolio value or return.

## Selection and limitations
The candidate was selected from 246 economic-hypothesis candidates, initially with EMA20. EMA40 followed a20/40/60 historical turnover comparison. All later historical intervals were inspected. Local27- and125-configuration grids support a performance plateau, not proof that all overfitting has been removed. Historical walk-forward simulation is not a newly untouched OOS experiment.

Earlier risk-style tests used public-method local CNE5 approximations, not licensed/full MSCI Barra. They found material style/known-signal overlap. The latest two-control study does not revoke those findings. Controls and smoothing differ across studies; improved residual results cannot be attributed solely to removing one style.

Gross results are prominent to inspect signal strength; the10bp companion and residual results remain visible to show implementation sensitivity. Positive IC does not establish statistically significant net portfolio returns. No direct sentiment measure or causal experiment is supplied. Current data end 2026-10-20; references to2025+ denote a historical later window, not pristine OOS.
