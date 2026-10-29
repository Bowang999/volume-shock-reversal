# Data interface

The repository contains **derived portfolio / IC / group statistics**, not stock-level market data. The empirical figures can be regenerated from these released aggregates. Synthetic data demonstrate the engine but do not reproduce empirical results.

## Three reproduction levels

1. **Figure reproduction:** run `python scripts/render_report.py`; released aggregate tables are sufficient.
2. **Engine demonstration:** generate the synthetic NPZ and run the CLI. Synthetic results have no financial interpretation.
3. **Independent market-data rerun:** provide aligned NPZ panels using the schema below. Exact empirical reproduction additionally requires the same provider history, adjustments, historical classifications and frozen daily universe, which are not redistributed here.

## NPZ schema

All arrays are T-by-N except `dates` (T, strictly increasing trading-session dates). Use `allow_pickle=False` compatible numeric/string dtypes.

| Field | Meaning |
|---|---|
| dates | Trading-session dates, datetime64[D] or ISO strings |
| close | Consistently adjusted positive marking closes; last known close carried on missing quote days |
| volume | Raw share volume, same share unit as float_shares |
| float_shares | Historical circulating shares known on the signal date |
| float_cap | Historical positive circulating market capitalization |
| industry | Historical industry integer code, -1 when unknown; encode categories compactly |
| observed | Whether that session's source close was actually observed |
| tradable | Original research execution eligibility; False freezes current positions |
| eligible (optional) | Frozen daily research membership, using current/past information only |

Signals require normal-trading, positive-volume observations. Strict windows are not filled across missing values. Portfolio marking uses carried closes; IC endpoints require observed entry/exit closes. `tradable` is an input, not an inferred full exchange execution model.

The published universe inherits earlier research availability constraints, and actual performance starts in2012. It is not a freshly reconstructed unrestricted all-A point-in-time universe. Source quote revisions, ST/listing/delisting screens, limits, borrow and capacity are not fully reconstructed. Do not mix unadjusted executions with adjusted signals/holdings.

Raw data are not licensed or redistributed by this repository. See reports/research_manifest.json for differences between the main, decile and parameter-audit accounting.
