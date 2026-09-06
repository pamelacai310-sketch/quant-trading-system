# September 7–12 native project-model paper trial

This supersedes the four manually selected instruments as the **main model test**.
The original `paper_week_20260907` SMA experiment remains unchanged and separately labelled.
No live orders or claims of a demonstrated edge are authorized by these results.

## Universe and decisions

- Freeze all 50 SSE50 constituents from the dated, linked public snapshot in `universe.json`.
  The pool is a reproducible large-cap segment, not the whole Chinese stock market and not a list of model recommendations.
- Also consider IF/IH/IC/IM September and December 2026 dated contracts. No continuous synthetic contract is executable.
- Fetch up to 252 completed unadjusted daily observations through the last completed session.
  Require 90 observations, current cutoff, positive finite prices/volume, unique dates and valid OHLC.
  Record individual provider failures; do not substitute example data or hand-selected symbols.
- Call the existing `SelfIteratingCausalEngine.run_learning_cycle` with these datasets, retaining the complete native report, feature/rejection reasons, scores and proposed actions.
- The model's score and causal labels remain research diagnostics: **neither is a calibrated expected net return or a proven causal effect**.
- Then apply ADV20 notional >= RMB50m, one contract per family by ADV20, native action requirement, top-five absolute native scores, stock long-only, native weight caps and lot/capital limits.
  Missing actions and engine errors mean cash, not a fallback SMA signal. Parameters remain frozen throughout the week.
- Log every candidate's data hash, cutoff, native action/score, cost budget, selection and rejection reasons in each decision event.

## Comparable accounts and execution

Both main candidate and same-pool SMA20 control start with RMB40m virtual capital.
RMB20m is split equally across 50 stock sleeves and RMB20m across eight futures sleeves.
At most 80% of each sleeve's NAV is used as notional, with no transfers or borrowing.
The control considers the five most liquid eligible instruments and applies SMA20 direction.
Both accounts have the same data, liquidity, cost, capital and execution gates; different realized exposures must be reported.
The old four-symbol trial has a different universe: do not interpret its return difference as isolated model alpha.

Orders can fill only against subsequently observed fresh quotes in the exchange's day sessions.
Maximum quote age 90 seconds, spread 5bps, displayed-book participation 10%,
ADV-based target cap 0.1%, declared round-trip cost budget 40bps including spread allowance.
Minimum fees and small-fill costs are checked at execution. These are declared conservative assumptions, not verified broker tariffs.
Stock T+1, ticks, lots, commissions, sell tax, adverse slippage and futures multiplier/margin apply.
A quote capture gap never authorizes retrospective fills. Cloud scheduler timing is not guaranteed.
The 2% drawdown gate queues exits on later executable quotes.
Friday permits exits only, to avoid entering T+1 positions that cannot close during the trial.
Unfilled exits remain marked open at the last recorded quote; never invent a Saturday liquidation.
These stock/index-futures instruments have no night session. The requested 02:00 cutoff is the evidence cutoff.

## Schedule and reporting

Asia/Shanghai: September 7 09:00 start, September 12 02:00 cutoff, 03:00 summary.
The GitHub workflow freezes the universe/data/code before start, then records chained immutable events.
If preparation misses the deadline, the main trial blocks instead of using a late retrospective bootstrap.
The separate SMA control remains operational.

Report net NAV profit/return, realized closed-cycle net profit, open positions and marks,
closed-trade count, net win rate, mean-win/absolute-mean-loss payoff, fills, missing data,
selection/rejection coverage, gross exposure, and a sensitivity subtracting another copy of recorded fees and adverse slippage.
That sensitivity is not a full doubled-spread/market-impact simulation.
Zero trades or an empty win/loss group produces null win rate/payoff where appropriate.
One week is a forward execution/selection test; it cannot establish repeatable advantage.
Corporate actions, capacity beyond displayed L1, calibrated edge estimation and fully nested model selection remain limitations.
