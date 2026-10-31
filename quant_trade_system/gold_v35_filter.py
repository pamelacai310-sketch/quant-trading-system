"""V3.5 P0 gold selling-pressure guard. Research only; no auto-trading."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Config:
    yield_drop_pp: float = .02
    min_gold_drop: float = 0.
    min_price_reclaim_days: int = 3
    max_flow_age_days: int = 7
    fee_per_side_cny: float = 15.
    slippage_cny_per_gram_per_side: float = .02
    au_contract_grams: int = 1000

def point_in_time_merge(market:pd.DataFrame, releases:pd.DataFrame, source:str, value_col:str)->pd.DataFrame:
    """Join on *release* timestamp, not CFTC Tuesday observation date."""
    if releases.empty:
        out=market.copy()
        out[source]=np.nan
        out[source+'_available_at']=pd.NaT
        return out
    for c in ('available_at',value_col):
        if c not in releases.columns: raise ValueError(source+' missing '+c)
    m=market.copy()
    m['cutoff_at']=pd.to_datetime(m['cutoff_at'],utc=True)
    e=releases[['available_at',value_col]].copy()
    e['available_at']=pd.to_datetime(e['available_at'],utc=True)
    if e['available_at'].isna().any() or e['available_at'].duplicated().any():
        raise ValueError(source+' invalid/duplicate release timestamps')
    merged=pd.merge_asof(m.sort_values('cutoff_at'),
                         e.sort_values('available_at'),left_on='cutoff_at',
                         right_on='available_at',direction='backward')
    if (merged['available_at']>merged['cutoff_at']).fillna(False).any():
        raise AssertionError('future CFTC/ETF data used')
    return merged.rename(columns={value_col:source,'available_at':source+'_available_at'}).sort_values('date').reset_index(drop=True)

def validate(m:pd.DataFrame)->pd.DataFrame:
    needed={'date','cutoff_at','next_au_open_at','gc_close','gc_high','gc_low','tips10_pp','dxy','au_next_open','au_next_close'}
    if needed-set(m.columns): raise ValueError('Missing columns '+str(sorted(needed-set(m.columns))))
    m=m.copy()
    m['date']=pd.to_datetime(m['date'])
    m['cutoff_at']=pd.to_datetime(m['cutoff_at'],utc=True)
    m['next_au_open_at']=pd.to_datetime(m['next_au_open_at'],utc=True)
    if m['date'].isna().any() or m['date'].duplicated().any(): raise ValueError('Invalid dates')
    m=m.sort_values('date').reset_index(drop=True)
    if m['cutoff_at'].isna().any() or m['cutoff_at'].duplicated().any():
        raise ValueError('Invalid signal cutoff')
    if (m['next_au_open_at']<=m['cutoff_at']).any():
        raise ValueError('Future info used: entry before cutoff')
    if (m['gc_close']<=0).any() or (m['tips10_pp'].abs()>20).any():
        raise ValueError('Gold price or TIPS scale invalid')
    if (m[['au_next_open','au_next_close']]<=0).any().any():
        raise ValueError('AU price invalid')
    return m

def backtest(market:pd.DataFrame,etf:pd.DataFrame,cot:pd.DataFrame,cfg:Config=Config()):
    """Compare rate+DXY baseline to V3.5 P0 veto, for next-session AU open->close.

    Inputs must be point-in-time verified: cutoff follows each US signal release,
    next_au_open_at follows cutoff; ETF/COT available_at means first publication.
    Any missing flow confirmation keeps the long-addition veto active.
    """
    m=validate(market)
    m=point_in_time_merge(m,etf,'etf_tonnes','etf_tonnes')
    m=point_in_time_merge(m,cot,'cot_mm_net','cot_mm_net')
    m['tips_delta_pp']=m['tips10_pp'].diff()
    m['dxy_delta']=m['dxy'].diff()
    m['gold_ret']=m['gc_close'].pct_change()
    m['baseline_long']=(m['tips_delta_pp']<=-cfg.yield_drop_pp)&(m['dxy_delta']<0)
    m['divergence']=(m['tips_delta_pp']<=-cfg.yield_drop_pp)&(m['gold_ret']<-cfg.min_gold_drop)
    for source in ('etf_tonnes','cot_mm_net'):
        m[source+'_delta']=m[source].diff()
        ts=source+'_available_at'
        m[source+'_new']=m[ts].ne(m[ts].shift())&m[ts].notna()
    m['positive_flow']=((m['etf_tonnes_new']&(m['etf_tonnes_delta']>0))|
                        (m['cot_mm_net_new']&(m['cot_mm_net_delta']>0)))
    active=False
    last_positive_flow_at=None
    gate=[];confirmed=[];exhaustion=[]
    for i in range(len(m)):
        row=m.iloc[i]
        if bool(row['divergence']):
            active=True
            last_positive_flow_at=None
        if active and bool(row['positive_flow']) and not bool(row['divergence']):
            last_positive_flow_at=row['cutoff_at']
        recent_flow=(last_positive_flow_at is not None and
                     (row['cutoff_at']-last_positive_flow_at).total_seconds()<=86400*cfg.max_flow_age_days)
        if i>=cfg.min_price_reclaim_days:
            prior=m.iloc[i-cfg.min_price_reclaim_days:i]
            price_reclaim=(float(row['gc_close'])>float(prior['gc_close'].max()) and
                           float(row['gc_close'])>float(prior['gc_close'].iloc[-1]))
            exhausted=(float(row['gc_close'])>=float(prior['gc_low'].min()) and
                       float(row['gc_close'])>float(prior['gc_close'].iloc[-1]))
        else:
            price_reclaim=False;exhausted=False
        release=(active and not bool(row['divergence']) and exhausted and
                 price_reclaim and recent_flow)
        if release:
            active=False;last_positive_flow_at=None
        gate.append(active);confirmed.append(bool(release));exhaustion.append(bool(exhausted))
    m['p0_block_new_long']=gate
    m['p0_release_confirmed']=confirmed
    m['selling_pressure_exhausted']=exhaustion
    m['v35_long']=m['baseline_long']&~m['p0_block_new_long']
    change=m['au_next_close']-m['au_next_open']
    cost=2*cfg.fee_per_side_cny+2*cfg.slippage_cny_per_gram_per_side*cfg.au_contract_grams
    for label,signal in (('baseline','baseline_long'),('v35','v35_long')):
        m[label+'_pnl_cny']=np.where(m[signal],change*cfg.au_contract_grams-cost,0.)
        m[label+'_correct']=np.where(m[signal],change>0,np.nan)
    def stats(label,signal):
        p=m.loc[m[signal],label+'_pnl_cny']
        pos=p[p>0];neg=p[p<0]
        cum=m[label+'_pnl_cny'].cumsum()
        return {'trades':int(m[signal].sum()),
                'direction_accuracy':float(m.loc[m[signal],label+'_correct'].mean()) if len(p) else None,
                'win_rate_net':float((p>0).mean()) if len(p) else None,
                'payoff_ratio':float(pos.mean()/abs(neg.mean())) if len(pos) and len(neg) else None,
                'profit_factor':float(pos.sum()/abs(neg.sum())) if len(pos) and len(neg) else None,
                'pnl_cny':float(p.sum()),
                'max_drawdown_cny':float((cum-cum.cummax().clip(lower=0)).min()) if len(cum) else 0.}
    report={'status':'RESEARCH_ONLY','observations':len(m),
            'divergences':int(m['divergence'].sum()),
            'blocked_baseline_longs':int((m['baseline_long']&m['p0_block_new_long']).sum()),
            'confirmed_releases':int(m['p0_release_confirmed'].sum()),
            'baseline':stats('baseline','baseline_long'),
            'v35':stats('v35','v35_long')}
    return m,report
