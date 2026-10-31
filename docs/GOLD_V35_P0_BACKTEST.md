# Gold V3.5 P0: falling TIPS yields while gold falls

**Status: implemented and unit-tested as research guard, not yet merged or live-deployed. 2024–2026 empirical backtest pending verified full input series.**

## Hypothesis and enforced behavior

When `DFII10` (10Y real yield, percentage points) declines by at least 0.02 (=2bp) from the last available observation **and** COMEX gold declines, the V3.5 P0 filter forbids **new long additions justified by lower real yields**. It does not force-close existing positions, does not generate a short, and does not automatically buy when the veto is lifted.

Veto release **requires BOTH**:
1. **Post-event price stabilization and confirmation**: latest GC close is above prior 3 completed trading days' highest close (and no renewed sell-pressure event).
2. **Freshly published fund-flow confirmation**: GLD ETF gold holdings increased, OR CFTC Gold Disaggregated Futures Only managed-money net longs increased; this observation must have become public after the sell-pressure event and within seven days of the confirmation.
Missing/stale flow => **FAIL CLOSED** on incremental longs. OI contraction alone never proves forced liquidation or exhaustion.

Initial thresholds are hypotheses, not learned values or validated optima.

## Source-data schema and publication times

`market.csv` columns:
`date,cutoff_at,next_au_open_at,gc_close,gc_high,gc_low,tips10_pp,dxy,au_next_open,au_next_close`

- `gc_close/high/low`: real COMEX GC tradable continuous contracts (roll method and timestamps documented); no synthetic data.
- `tips10_pp`: FRED `DFII10`, percent units e.g. 2.63 (not .0263), and **published** by the chosen cutoff.
- `dxy`: actual ICE US Dollar Index. The Fed broad dollar index is not DXY.
- `au_next_open/close`: executable next China session after the signal cutoff (AU main rolling contract for 2024–2025; **AU2612 ONLY after actual listing**, independent test). Respect Chinese holidays and COMEX holidays.
- `cutoff_at`: ISO UTC instant when ALL market signals were available; `next_au_open_at`: first feasible execution after cutoff. No `next_au_open_at <= cutoff_at`.

`etf.csv`: `available_at,etf_tonnes`, GLD custody history based on **actual release availability**, not the day ETF traded. GLD is a proxy for gold ETF flows, not worldwide aggregate flows.

`cot.csv`: `available_at,cot_mm_net`, net managed-money GOLD futures-only long-minus-short from CFTC. Tuesday observation dates are NOT availability. CFTC normally releases Tuesday positions on Friday 3:30pm ET, adjusted for holidays. Provide the actual UTC release timestamps.

Authoritative starting points:
- FRED TIPS: https://fred.stlouisfed.org/series/DFII10
- Official CFTC disaggregated archives: https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm
- CFTC release schedule: https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm
- SPDR GLD daily historical archive: https://www.spdrgoldshares.com/usa/gld/
- CME tradable GC and rollover data: https://www.cmegroup.com/market-data/cme-group-continuous-price-series.html

### Mandatory provenance check

Existing `data/gold_daily.csv` in this repository covers only **2022-01-03 to 2024-10-04**, and on **2024-10-04 it claims 1113.1258** when real COMEX Dec-2024 gold settled around **2667.8 USD/oz**. It is NOT valid research COMEX history. Do not use or relabel it as actual market data.

## Run research ablation

```bash
pip install pandas numpy
python -m unittest tests.test_gold_v35_filter -v
python scripts/run_gold_v35_backtest.py --market market.csv --etf etf.csv --cot cot.csv --out results/gold_v35
```

`latest_p0_status(history, etf, cot)` can also be called to check the live long-addition veto from historical market inputs alone (requires no future AU label).

### Metrics and interpretation

Compare **same baseline** (next-session AU long if previous observable DFII10 and DXY both fell) versus the V3.5-filtered strategy:

- Signal count and number of vetoed entries; direction hit-rate on trades actually taken; net win rate, payoff ratio (mean net win / mean absolute net loss), profit factor, total PnL and maximum drawdown.
- Separate 2024, 2025 and 2026 results. Use 2024 to develop, 2025 to validate and 2026 as held-out test. Do not tune on 2026.
- Weekly-block bootstrapped 95% uncertainty interval of *incremental PnL*. A CI crossing zero is not convincing evidence of improvement.
- Defaults are **illustrative**, not actual broker fees: 1 AU lot (=1000g); CNY15 commission per side and CNY0.02/g slippage per side. Update actual fees, slippage, margin, rollover and instruments before relying on results.
- Backtest checks early-2024 to Oct-2026 coverage, >=120 records/year, and declines incomplete inputs; this does not replace independent validation of dataset authenticity.

### Important limits

This is an **ablation on the simple falling-real-yield / falling-DXY long signal**, not a claim that the whole news→JEV→Codex→AU2612 system is already improved. To measure whole-system incremental value, also run shadow mode against timestamped historical V3.4 signals with otherwise identical execution and risk controls.

This PR neither calls JEV nor contains secret keys. JEV can annotate news context but the P0 veto itself should be deterministic, testable code.


## 调用示例：未来每次黄金预测的P0门控

```python
from quant_trade_system.gold_v35_filter import (
    latest_p0_status, enforce_new_long_target, constrain_rate_upside_increment
)

# historical_market 为截止预测时点已知的 COMEX、TIPS、DXY 数据。
# published_etf / published_cot 都必须含真实公开时刻 available_at。
guard = latest_p0_status(historical_market, published_etf, published_cot)
proposed_long_probability_delta = constrain_rate_upside_increment(
    delta_probability_attributable_to_lower_real_yields, guard
)
approved_position, reason = enforce_new_long_target(
    current_position, strategy_proposed_position, guard
)
# 上层引擎仍需将概率重新规范化、记录reason并实施其它风控。
```

**注意**：这里提供调用API，不代表已自动连接所有现有策略或实时执行引擎；需显式从预测/下单入口接线。缺少ETF/CFTC数据时保守地禁止仅凭实际利率下降新增多头。价格/资金流解锁并不生成买入信号。
