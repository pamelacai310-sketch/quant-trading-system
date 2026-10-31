"""Research-only 2024–2026 gold P0 ablation; requires verified market data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from quant_trade_system.gold_v35_filter import backtest

def summarize_by_year(frame):
    result={}
    for year,g in frame.groupby(frame['date'].dt.year):
        result[str(year)]={}
        for label in ('baseline','v35'):
            selected=g.loc[g[label+'_long']]
            pnl=selected[label+'_pnl_cny']
            plus=pnl[pnl>0];minus=pnl[pnl<0]
            result[str(year)][label]={
                'trades':len(selected),
                'hit_rate':float(selected[label+'_correct'].mean()) if len(selected) else None,
                'net_pnl_cny':float(pnl.sum()),
                'payoff_ratio':float(plus.mean()/abs(minus.mean())) if len(plus) and len(minus) else None,
                'profit_factor':float(plus.sum()/abs(minus.sum())) if len(plus) and len(minus) else None}
        result[str(year)]['incremental_pnl_cny']=float(
            (g['v35_pnl_cny']-g['baseline_pnl_cny']).sum())
    return result

def bootstrap_by_week(frame,n=1000,seed=35):
    weeks=frame['date'].dt.to_period('W-SUN')
    diff=frame['v35_pnl_cny']-frame['baseline_pnl_cny']
    week_pnl=np.array([float(diff.loc[weeks==w].sum()) for w in weeks.unique()])
    if len(week_pnl)<10:return {'status':'INSUFFICIENT_WEEKS'}
    rng=np.random.default_rng(seed)
    resamples=rng.choice(week_pnl,size=(n,len(week_pnl)),replace=True).sum(axis=1)
    return {'observed_incremental_net_pnl_cny':float(week_pnl.sum()),
            'ci95_week_bootstrap_cny':[float(x) for x in np.quantile(resamples,[.025,.975])],
            'fraction_bootstrap_positive':float((resamples>0).mean()),
            'weekly_blocks':len(week_pnl)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--market',required=True)
    p.add_argument('--etf',required=True)
    p.add_argument('--cot',required=True)
    p.add_argument('--out',default='results/gold_v35')
    args=p.parse_args()
    paths=[Path(v) for v in (args.market,args.etf,args.cot)]
    if not all(x.is_file() for x in paths):
        p.error('Missing source files: must provide verified COMEX, AU, TIPS, DXY, GLD ETF, CFTC data')
    market,etf,cot=[pd.read_csv(f) for f in paths]
    if any(x.empty for x in (market,etf,cot)):
        p.error('Empty source: no synthetic backtest substitution is permitted')
    frame,report=backtest(market,etf,cot)
    counts=frame['date'].dt.year.value_counts().to_dict()
    if (frame['date'].min()>pd.Timestamp('2024-01-31') or
        frame['date'].max()<pd.Timestamp('2026-10-01') or
        any(counts.get(y,0)<120 for y in (2024,2025,2026))):
        raise SystemExit(f'INCOMPLETE_2024_2026: year_counts={counts}. No 3-year results.')
    report['by_year']=summarize_by_year(frame)
    report['weekly_block_bootstrap']=bootstrap_by_week(frame)
    report['caveat']='Research baseline only, 1 lot per signal; no deployment or live execution claims.'
    target=Path(args.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(str(target)+'.csv',index=False)
    Path(str(target)+'.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
