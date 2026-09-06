# P0 repair and September 7–11 forward validation

Base: `9c9d3d3b602e6bace4dd2fdb1ec9f3fff5980a28`. All times below are Asia/Shanghai.

## What changed

| Defect | Resolution | Boundary |
|---|---|---|
| Direction correctness changed the sign of investment returns | Shared `position_returns`: position × explicitly aligned forward return − turnover costs | Classifier accuracy is separate; closed-trade win rate and odds are unavailable without a trade ledger |
| Same-close signal and fill | Completed prior-bar decisions execute at next open, adverse slippage and fees | Bar execution is an estimate; no queue/partial-fill/market-rule certification |
| Rebalancing reset entry prices and distorted short capital | Shared average-cost ledger, proportional entry-fee allocation; fixed entry units until exit | Full stock and futures exchange execution remains out of scope |
| Unrealized marks counted as winning closed trades | Open positions remain separate; closed PnL includes both fees | Marked NAV includes unrealized PnL; reserved liquidation cost is separately reported |
| Pre-entry or post-exit OHLC contaminated MAE/MFE | Start excursion at fill; never use exit bar's later high/low | Daily OHLC cannot reveal intraday event ordering |
| Non-differentiable win/odds losses and absolute-return incentive | Transformer trains proper BCE; misleading odds/elasticity losses explicitly reject use | These models estimate direction probability, not a demonstrated return-optimal policy |
| Sequence length and checkpoint mismatches | Persist training sequence length and restore best validation checkpoint | Validation is tuning data, not independent final evidence |
| Annual Sharpe inserted into per-period inference | Per-period inference, annual display; unknown trial dispersion cannot pass multitrial DSR | IID assumption remains; dependent data need independent block inference |
| Precomputed-return CPCV presented as model validation | Diagnostic scope explicit; cannot pass promotion | Fold-local refitting is P1 work |
| Broken `_t-12`/underscored variables and silent formulas | Whitelisted AST arithmetic, exact lags/windows, visible feature errors, no all-NaN factors | Unsupported formulas fail explicitly; no arbitrary Python evaluation |
| Cross-symbol raw-price averaging | Single-symbol matrices or explicit per-symbol panels | Callers with multiple symbols must use `generate_feature_panels` |

No changes assert that causal identification, CVaR/pairs placeholder modules or broker live-fill reconciliation are complete. The nightly heuristic engine is not replaced by this experiment. P0 removes false accounting/training evidence; it does not demonstrate alpha.

## Verification

- Existing discovery run: 198 tests passed (147.6 seconds).
- Final targeted accounting, forward evidence, adaptive risk and robustness run: 18 tests passed; PyTorch gradient test executed, not skipped.
- Two-epoch small Transformer training and inference with sequence length 4 passed.
- Compilation and `git diff --check` passed.
- Synthetic prices appear only in correctness tests. They are never forward performance observations.

Run:

```bash
python -m unittest tests.test_p0_accounting tests.test_forward_evidence tests.test_backtest_adaptive_risk tests.test_robustness_controls -v
python -m quant_trade_system.forward_evidence capture
python -m quant_trade_system.forward_evidence report
```

## Frozen forward protocol

The authoritative manifest is `experiments/p0_week_20260907/protocol.json`, with source/protocol hashes in `lock.json`. A real-data bootstrap for September 4 is archived before the September 7 opening. Failed data attempts are retained. Observations use exclusive creation and a hash chain; the report rejects a broken chain. Git commits provide an external chronological record; local timestamps alone are not trusted third-party attestations.

- Window: September 7–11, 2026, five day sessions. CFFEX IF2612/IM2612 have been chosen explicitly; this experiment does not cover commodity night sessions or continuous-contract pseudo-fills.
- Universe: 600519, 601318, IF2612, IM2612. No replacement of losing or unavailable instruments within the window.
- Candidate: frozen SMA20 rule; stock long/flat, futures long/short. Flat before reversing; no same-day stock round trip.
- Baseline: fixed long exposure, same sleeve capital/notional limit; cash return reported as zero comparator. Equal limits do not imply identical realized volatility or beta.
- Capital: four independent RMB 10 million shadow sleeves, maximum 20% notional allocation per sleeve. These are simulated account sizes, not user capital or orders.
- Orders are stored after a completed close, before the next session. Next-day fills use that previously committed quantity and the next opening price. No outcome-dependent retuning.
- Stock lots: 100 shares; specific futures contract multiplier and integer lots. Slippage: 5 bps rounded adversely to tick. Stock fee 3 bps, minimum RMB 5; sell tax assumption 5 bps. Futures fee assumption 1 bp. All are disclosed research assumptions, not verified broker costs.
- Public unadjusted Sina data via AKShare. Volume, freshness, timestamps and OHLC checks are mandatory; one-price bars, large gaps, participation violations or missing data invalidate the run. Public daily bars cannot establish actual fill availability or verify corporate actions.
- Margin assumption: 20% for the futures shadow sleeves, with conservative notional allocation; breach rejects the run. No broker margin certification or intraday liquidation claim.
- Keep complete raw inputs, source/version, positions, fills, fees, net realized PnL, marked NAV, reserved exit costs and next orders. Friday open positions are not fabricated as closed winners.
- Report candidate/baseline net marked returns and excess, closed-trade wins/payoff where defined, drawdown and fixed-order doubled-incurred-cost sensitivity. Losing outcomes are retained. No annualized week return or purported confidence interval from five dependent daily observations.
- The predeclared minimum for even considering inference is 30 independent days. Five daily observations cannot satisfy it. Minimum sample count alone would also not prove reproducibility: independent future windows, fitted-model holdouts, block uncertainty, cost verification and execution reconciliation are still necessary.

## Scheduling and outcome

`P0 frozen forward evidence` runs at 18:17, 19:17 and 20:17 during the finite September 6–11 window; successful daily captures are idempotent. Only the first successful attempt writes observations. Retries preserve failures. GitHub scheduling can be delayed; it is not an exchange-connected intraday service. No process sends real orders.

The workflow must exist on the default branch to receive scheduled runs. It writes evidence back to the repository and uploads an artifact. If the write is blocked, the artifact/log remains reviewable, and the next run must not reconstruct an uncommitted order as if it had existed before the open.

Possible statements after this week: **observed net gain**, **observed net loss**, **outperformed/underperformed this baseline**, or **data/sample insufficient**. None is equivalent to “repeatable net-profit advantage proven.” The report's production and repeatability flags remain false.

Calendar references checked September 5: [SSE 2026 holidays](https://www.sse.com.cn/disclosure/dealinstruc/closed/), [SHFE 2026 holidays](https://www.shfe.com.cn/services/calenderandholidays/holiday/). Data interfaces: [AKShare futures documentation](https://akshare.akfamily.xyz/data/futures/futures.html), [AKShare stock documentation](https://akshare.akfamily.xyz/data/stock/stock.html).
