# Live-quote paper trading, September 7–12, 2026

User authorization: publish P0 source/workflows/evidence, merge, and run paper trading from **2026-09-07 09:00 Asia/Shanghai** through **2026-09-12 02:00**, with the account summary at **03:00**. No real-money order is authorized or transmitted.

This protocol supersedes the prior daily-bar execution schedule, whose cron has been removed. Its original September 4 data bootstrap is retained as provenance. New source/protocol hashes and the bootstrap hash are frozen under `experiments/paper_week_20260907/lock.json` before start. Do not change the strategy within the evaluation period.

## Model and execution

- The frozen model is the **SMA20 daily candidate from the prior P0 experiment**, not a fitted Transformer and not the entire nightly causal system. Stocks are long/flat; futures long/short; reversals flatten before reopening. The fixed-long baseline uses separate, equally funded shadow accounts.
- Universe: 600519, 601318, IF2612, IM2612. Two A shares and two specific CFFEX contracts; no commodity night sessions. 09:00 starts order preparation. Executions require an open market and fresh quotes; no synthetic fills at 02:00 Saturday.
- Four RMB 10 million virtual sleeves per candidate/baseline, each with at most 20% notional at entry. These amounts are simulation inputs, not the user's capital.
- After completed daily data are available, call the same `make_orders` model; persist targets and creation time. Check orders on fresh displayed bid/ask approximately every five minutes during day sessions.
- Quote time must be strictly later than order creation, no more than 90 seconds old and not in the future. A failed/missed poll is never reconstructed as a historical execution.
- Fill only displayed best-side quantity, round to 100-share lots or integer contracts, permit partial fills, prevent reuse of a quote for the same account/instrument and enforce stock T+1. One snapshot may serve the independent candidate and baseline simulations.
- Declared execution assumptions: adverse tick-rounded 5 bps beyond displayed bid/ask; stock commission 3 bps, minimum RMB 5 **plus** 5 bps sell tax; futures commission 1 bp. Futures multiplier IF=300, IM=200; conservative margin assumption 20%. These are not verified broker tariffs or actual queue fills.
- The shared ledger allocates entry fees proportionally when closing partially. Win rate and payoff ratio count **complete flat-to-flat cycles**, not child orders or classifier hits.
- At the cutoff expire pending orders. Mark remaining positions at the last observed valid quote or archived closing valuation, showing its timestamp. No forced after-hours execution. Net NAV includes these unrealized gains/losses; open positions do not count as winning closed trades.
- A 2% sleeve-account aggregate drawdown trigger halts additions and queues exits; execution still requires valid subsequent trading quotes, stock settlement and sufficient market size. This is not an intraday guaranteed stop price.

## Outputs

`events/000000.json` onward preserves the hash chain, raw quotes, model inputs, orders, partial fills, fees, positions and closed cycles. `summary.json` reports candidate and baseline separately:

- net profit and net return from the initial virtual cash;
- complete closed-trade count and net win rate;
- payoff ratio = average positive net cycle PnL / absolute average negative net cycle PnL;
- realized closed-cycle PnL and open positions/valuation timestamps;
- actual start, observed quote days, failures, pending/expired orders.

If there are no closed cycles, win rate is `null`. If either winning or losing samples are absent, payoff ratio is `null`, rather than a fabricated 0, infinity or advertised advantage. Five days cannot prove repeatable alpha.

## Operations

```bash
python -m quant_trade_system.paper_week step
python -m quant_trade_system.paper_week report
```

The `Paper trading September 7-12` GitHub workflow runs during the finite week, commits every ledger step before later runs consume its orders, and uploads a recovery artifact. Repository write conflicts fail rather than force-overwrite history. Scheduled infrastructure can start late or skip queued jobs; report actual coverage. The start, cutoff and summary are also checked by ChatGPT scheduled tasks. Never replace a missed signal/quote with a backdated order.

Local regression covers fresh/later quotes, stale/future rejection, repeated-quote prevention, partial fills, T+1, full-cycle fee reconciliation, cutoff and no after-hours fills. Real-data preflight verifies all four quote schemas without executing weekend quotes.
