# December contract forecasting repair

The September 10 audit found no established historical edge in independent
seven-feature regressions. The replacement research entry point partially pools
contracts, uses separate close and settlement targets, and counts validation
evidence in trading dates. A frozen night-price reference is explicitly distinct
from an estimate of the remaining session's return.

Run `python scripts/run_structural_forecast.py --input input.json --output new.json`.
The output must not already exist. The native causal engine also exposes
`run_structural_daily_research`; its legacy five-row causal score is not a daily
percentage prediction. `daily_forecast.py` is retained for paired baseline replay.

## Repaired mechanisms

- Shared market and sector inputs plus penalized sector/contract intercepts.
  Training weights give each date total weight independent of its contract count;
  sectors split that date weight equally. Returns are normalized by past 20-day
  volatility. Train-only imputation/scaling prevents future information leakage.
- Holding, turnover, volume drift, active-contract age, and days to the delivery
  month describe lifecycle changes. The last item is not an exact last-trade date.
  The old model already had a volume/MA20 feature; it did not lack all volume data.
- November/January basis and relative volume plus a vendor main-contract proxy.
  Continuous main-contract returns are not used because roll gaps may contaminate
  them. Missing inputs have separate indicators; no fictitious quotes are filled.
- Overseas returns enter only at explicit conservative availability times. News
  archives require time-zone-aware release times and have age limits. Reconstructed
  archives remain retrospective evidence and are not immutable forward records.
- Separate `close[t+1]/close[t]-1` and `settle[t+1]/settle[t]-1` models. A quote's
  last trade divided by previous settlement is a third, distinct quantity.
  The settlement gate must beat BOTH unchanged settlement and tomorrow's
  settlement equal to today's close. This prevents mechanical settlement lag
  from being mistaken for predictive skill.
  Settlement regressions learn the residual relative to the current-close
  anchor and shrink only that residual, not the already-known close/settle gap.
  Directional advantage is also measured against the anchor's direction where
  it makes a nonzero call; neutral-anchor comparisons use 50% as the hurdle.
- Contract, sector, market-family and sector/regime normalized-residual intervals
  use only previously realized labels. Each date has equal calibration mass.
  The conservative envelope of available strata is widened 25% for OpenClaw risk
  >=6. This rule is a disclosed risk assumption, not an empirically optimal choice.
- Aggregate date losses precede a circular five-day block bootstrap. Multiple
  comparisons cover both targets, both candidate models, and sector/overall
  screens. No inference treats dozens of correlated contracts as independent.
- A midnight-aware night snapshot requires a matching previous settlement. Its
  expected remaining drift is explicitly zero absent validated intraday alpha.
  A scenario envelope is not labeled a calibrated 80% intraday interval.
- No account fees, slippage, or order-book history means no after-cost return is
  invented. All study outputs retain `can_trade=false` and `forward_validated=false`.

## Prespecified research policy and honest evaluation

Minimum training is 60 distinct sessions; maximum is 120 with 60-session half-life.
At least 20 eligible observations of a contract are needed. Ridge penalty is 30
and prediction shrinkage is 0.5. Two fixed candidates (core and enriched) are
reported without selecting the most successful model on the evaluation sample.
The gate uses 60 distinct outcome dates, at least 60 simultaneous comparisons for
the present 14-sector pool, and 20,000 bootstrap draws. Interval calibration needs
30 distinct dates. These are design choices, not hyperparameters claimed optimal.

The architecture was developed after examining September 8 and 10. All historical
replays are development diagnostics, even when every row observes temporal
separation. They are not untouched holdouts. Replay compares old/new on identical
contract-date pairs and reports interval coverage by sector. The current cohort
is frozen at the requested forecast date, so no survivor-free strategy backtest
is claimed. Newly timestamped frozen outputs start future evaluation separately.

Irreducible innovations, uncollected inventory vintages, incomplete execution
history, vendor revisions and future predictive edge cannot be repaired by code
alone. They remain explicit limits, with no promotion to live execution.
