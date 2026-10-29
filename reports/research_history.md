# Research history

1. Economic hypothesis: abnormal-volume price changes may contain transient buying/selling pressure and later reversal.
2. Candidate construction and selection: ECO0036 emerged from 246 candidates; its first smoothing span was20. Original training direction selection was retained.
3. Unified daily engine: rank -> industry/float-cap joint OLS -> standardized residual; linear+100%/-100%; next-close execution; IC/Rank IC and actual deciles.
4. EMA20/40/60 comparison: EMA40 reduced one-way turnover roughly30%-32% against EMA20; chosen after inspecting history.
5. Earlier style exposures and existing-factor checks: local CNE5 approximations, actual holding-weighted exposures, correlations and fixed combinations with Alpha095/Alpha132/CorrPV/extreme-return/reversal signals. Material overlaps remain.
6. Broad27-parameter and dense125-parameter audits, plus historical market subperiods. No replacement of5/5/40 with the best grid cell.
7. Universe/execution stresses: omit bottom20% float cap, omit bottom20% prior-average amount, add one execution session delay; paired baselines retained.
8. Prior-three-year annual parameter selection with a21-session gap; CSCV/local block bootstrap. Local parameter-ranking PBO around38%-41% is not a future-loss probability or global overfit correction.
9. Five-by-five conditional sorting within pure reversal bins: all five historical groups monotone through2024; four strict groups in the later window, with some differences insignificant.
10. Latest focused controls: same-EMA pure reversal and abnormal-turnover attention proxy, individually and jointly; no RV. All five full10bp ledgers matched an independent Python reference for3550 sessions.
11. Future protocol recorded: freeze parameters and assess genuinely new data after a preset observation interval. No completed forward results or live trading are claimed.

The local project contains more historical trials than the focused release. A positive local audit does not repair the lack of a complete independent-trial count. Formula, cost assumptions and the publicly shown intervals must not be retrospectively described as preregistered.
