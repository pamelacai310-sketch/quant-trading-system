"""Execute real V3.5 P0 ablations using auditable 2024–2026 research inputs."""
from __future__ import annotations
import json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"quant_trade_system"))
sys.path.insert(0,str(ROOT/"scripts"))
from gold_v35_filter import backtest
from run_gold_v35_backtest import summarize_by_year,bootstrap_by_week

def one(name,path,etf,cot):
    raw=pd.read_csv(path)
    if len(raw)<110:raise ValueError('Too few genuine input observations '+name)
    frame,report=backtest(raw,etf,cot)
    report['input_rows']=len(raw)
    report['period_start']=str(frame['date'].iloc[0].date())
    report['period_end']=str(frame['date'].iloc[-1].date())
    report['by_year']=summarize_by_year(frame)
    report['incremental_v35_net_pnl_cny']=report['v35']['pnl_cny']-report['baseline']['pnl_cny']
    report['weekly_block_bootstrap']=bootstrap_by_week(frame,n=4000,seed=35)
    report['v35_evaluation_basis']='actual dated COMEX GC=F, FRED TIPS, DXY, SPDR GLD, CFTC and EastMoney AU prices; backtest uses conservative lagged release dates'
    report['instrument']='AU2612 actual fixed maturity contract' if name=='AU2612' else 'AU main continuous synthetic instrument (proxy)'
    report['caveats']=[
        'Does NOT prove predictive accuracy of the full JEV/Codex news model',
        'Uses 1 lot per signal open->close, illustrative fee/slippage; actual broker fills differ',
        'Main continuous is NOT a tradable single contract and may have roll artifacts',
        'Release lags are CONSERVATIVE; actual exchange session open timestamps require spot check',
        'COMEX OI is weekly last-published CFTC aggregate, not CME contract daily OI',
        'Not out-of-sample parameter tuning if thresholds were chosen after 2024-2026 analysis'
    ]
    print('REAL_BACKTEST_RESULT',name,json.dumps(report,ensure_ascii=False))
    return report,frame

def main():
    research=ROOT/'data'/'research_inputs'
    etf=pd.read_csv(research/'gld_releases_LOCAL_ONLY.csv')
    cot=pd.read_csv(research/'cftc_releases.csv')
    outcomes={}
    resultdir=ROOT/'results'
    resultdir.mkdir(exist_ok=True)
    targets={
        'AU2612':research/'gold_v35_au2612_market.csv',
        'AU_MAIN':research/'gold_v35_au_main_market.csv'}
    for name,path in targets.items():
        if not path.is_file():
            outcomes[name]={'status':'NOT_RUN','reason':'No verified source for this instrument'}
            continue
        report,frame=one(name,path,etf,cot)
        outcomes[name]=report
        frame[['date','divergence','p0_block_new_long','p0_release_confirmed','baseline_long','v35_long','baseline_pnl_cny','v35_pnl_cny']].to_csv(resultdir/('gold_v35_signals_'+name+'.csv'),index=False)
    outcomes['status']='RESEARCH_REAL_DATA'
    outcomes['data_asof']='2026-10-30'
    outcomes['input_provenance']=json.loads((research/'provenance.json').read_text(encoding='utf-8'))
    (resultdir/'gold_v35_real_backtest_20261030.json').write_text(json.dumps(outcomes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('BACKTEST_SUMMARY',json.dumps({k:{'rows':v.get('input_rows'),'baseline':v.get('baseline'),'v35':v.get('v35'),'incremental':v.get('incremental_v35_net_pnl_cny')} for k,v in outcomes.items() if isinstance(v,dict)},ensure_ascii=False))
if __name__=='__main__':main()
