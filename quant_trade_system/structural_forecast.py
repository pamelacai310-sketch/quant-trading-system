"""Time ordered, partially pooled December-contract research and live references.

No broker orders. Model development used pre-cutoff outcomes, so a historical
replay is diagnostic; only a newly frozen snapshot can start forward evidence.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import re
import numpy as np
import pandas as pd
from .daily_forecast import SECTORS, align_bars, local_features

SECTORS = {**SECTORS, 'UR':'chemicals', 'LC':'battery_materials',
           'EC':'shipping', 'LG':'forestry', 'CY':'textiles', 'BB':'paper_and_board',
           'WR':'steel', 'ZC':'energy'}

@dataclass(frozen=True)
class StructuralPolicy:
    min_train_days: int = 60
    max_train_days: int = 120
    min_contract_rows: int = 20
    ridge_alpha: float = 30.
    shrinkage: float = .5
    half_life_days: float = 60.
    calibration_days: int = 60
    min_calibration_dates: int = 30
    interval_mass: float = .8
    gate_days: int = 60
    block_days: int = 5
    bootstrap_samples: int = 20000
    family_tests: int = 60
    error_alpha: float = .05

def product(symbol): return re.sub(r'\d+$','',symbol).upper()
def sector(symbol): return SECTORS.get(product(symbol),'unknown')

def point_in_time(events, dates, column, max_age_days=4):
    """Calendar dates alone are insufficient: require explicit availability time."""
    result=[]
    for day in dates:
        cutoff=pd.Timestamp(day+'T15:30:00+08:00')
        eligible=[]
        for event in events:
            if not event.get('available_at') or event.get(column) is None: continue
            at=pd.Timestamp(event['available_at'])
            if at.tzinfo is None: raise ValueError('Availability must include timezone')
            if at<=cutoff and (cutoff-at).total_seconds()<=max_age_days*86400:
                eligible.append((at,float(event[column])))
        result.append(max(eligible,key=lambda x:x[0])[1] if eligible else np.nan)
    return pd.Series(result,index=dates,dtype=float)

def auxiliary_series(rows, dates, field='close'):
    if not rows: return pd.Series(np.nan,index=dates,dtype=float)
    frame=pd.DataFrame(rows)
    if frame.date.duplicated().any(): raise ValueError('Duplicate auxiliary date')
    return pd.to_numeric(frame.set_index('date').get(field),errors='coerce').reindex(dates)

def build_panel(bars, calendar, asof, auxiliary=None, external_events=None, news=None):
    if asof not in calendar: raise ValueError('asof absent from calendar')
    dates=[d for d in calendar if d<=asof]
    frames={s:align_bars(f,calendar,asof).reindex(dates) for s,f in bars.items()}
    returns=pd.DataFrame({s:f.close.pct_change(fill_method=None) for s,f in frames.items()})
    auxiliary=auxiliary or {}; external_events=external_events or {}; news=news or []
    outside={k:point_in_time(v,dates,'return',max_age_days=5) for k,v in external_events.items()}
    news_values={k:point_in_time(news,dates,k,max_age_days=2) for k in ['risk','energy_mentions','rates_mentions','trade_mentions']}
    allrows=[]
    for s,f in frames.items():
        p=product(s); sec=sector(s)
        family='equity' if sec=='equity' else 'bonds' if sec=='bonds' else 'commodity'
        x=local_features(f).add_prefix('local_')
        peers=[q for q in frames if q!=s and sector(q)==sec]
        market=[q for q in frames if q!=s and
                ('equity' if sector(q)=='equity' else 'bonds' if sector(q)=='bonds' else 'commodity')==family]
        for name,subset in [('market',market),('sector',peers)]:
            peer=returns[subset]; count=peer.notna().sum(axis=1)
            x[name+'_return']=peer.mean(axis=1).where(count>0)
            x[name+'_return5']=x[name+'_return'].rolling(5,min_periods=3).mean()
            x[name+'_breadth']=((peer>0).sum(axis=1)/count).where(count>0)
            x[name+'_dispersion']=peer.std(axis=1,ddof=0).where(count>0)
            x[name+'_count']=count
        # Months are calendar contracts; this is not a claimed exact last-trade date.
        x['life_days_to_delivery_month']=[(pd.Timestamp('2026-12-01')-pd.Timestamp(d)).days for d in dates]
        oi=pd.to_numeric(f.get('hold'),errors='coerce')
        x['life_log_oi']=np.log1p(oi.where(oi>0))
        x['life_oi_change5']=oi.pct_change(5,fill_method=None)
        x['life_turnover']=f.volume/oi.where(oi>0)
        x['life_volume_ratio']=f.volume.rolling(5).mean()/f.volume.rolling(20).mean()
        x['life_volume_drift']=np.log(f.volume.rolling(20).mean()/f.volume.rolling(60,min_periods=30).mean())
        x['life_age_sessions']=f.close.notna().cumsum()
        x['micro_close_vs_settle']=f.close/pd.to_numeric(f.get('settle'),errors='coerce').where(lambda z:z>0)-1
        for suffix,name in [('2611','near'),('2701','far'),('0','main')]:
            rows=auxiliary.get(p+suffix,[])
            other=auxiliary_series(rows,dates)
            x['curve_'+name+'_basis']=f.close/other.where(other>0)-1
            other_vol=auxiliary_series(rows,dates,'volume')
            x['curve_'+name+'_volume_share']=f.volume/(f.volume+other_vol.where(other_vol>0))
            # pct_change on a continuous main series could include a roll gap;
            # only same-date basis and volume share are used, labeled as vendor proxy.
        for k,v in outside.items(): x['external_'+k]=v
        for k,v in news_values.items(): x['news_'+k]=v
        x=x.replace([np.inf,-np.inf],np.nan)
        x['symbol']=s; x['sector']=sec; x['family']=family; x['feature_date']=dates
        x['target_date']=[dates[i+1] if i+1<len(dates) else None for i in range(len(dates))]
        for target,field in [('close','close'),('settle','settle')]:
            price=pd.to_numeric(f[field],errors='coerce').where(lambda z:z>0)
            # A zero-volume or invalid OHLC day must not supply settlement labels either.
            price=price.where(f.close.notna())
            r=price.pct_change(fill_method=None)
            x['y_'+target]=price.shift(-1)/price-1
            x['sigma_'+target]=r.rolling(20,min_periods=15).std(ddof=1)
        x=x.loc[f.close.notna() & (f.volume>0)]
        allrows.append(x)
    return pd.concat(allrows,ignore_index=True),frames

def weighted_fit(x,y,q,weights,alpha):
    """Median imputation and scaling fit ONLY on the current training window."""
    finite=np.isfinite(x)
    med=np.array([np.median(x[finite[:,j],j]) if finite[:,j].any() else 0. for j in range(x.shape[1])])
    xt=np.where(finite,x,med); qt=np.where(np.isfinite(q),q,med)
    w=weights/weights.sum(); mean=(xt*w[:,None]).sum(axis=0)
    sd=np.sqrt(((xt-mean)**2*w[:,None]).sum(axis=0)); sd=np.where(sd>1e-10,sd,1.)
    z=np.clip((xt-mean)/sd,-5,5); zq=np.clip((qt-mean)/sd,-5,5)
    center=(z*w[:,None]).sum(axis=0); ym=float(y@w)
    z-=center; zq-=center
    # Normalize to number of dates rather than number of correlated contracts.
    a=z.T@(z*weights[:,None])+alpha*np.eye(z.shape[1])
    beta=np.linalg.solve(a,z.T@((y-ym)*weights))
    contributions=zq*beta
    return ym+contributions.sum(axis=1),contributions,ym,{'medians':med.tolist(),'means':mean.tolist(),'scales':sd.tolist(),'coefficients':beta.tolist()}

def clustered_gate(records,policy,asof=None):
    if not records: return {'passed':False,'reason':'no_oos','dates':0}
    f=pd.DataFrame(records)
    if asof: f=f.loc[f.target_date<=asof]
    f=f.loc[np.isfinite(f.actual)&np.isfinite(f.prediction)].copy()
    f['improvement']=abs(f.actual)-abs(f.actual-f.prediction)
    f['anchor_improvement']=abs(f.actual-f.get('anchor_prediction',0.))-abs(f.actual-f.prediction)
    anchor=f.get('anchor_prediction',0.)
    anchor_hit=np.where(np.asarray(anchor)!=0,(np.sign(f.actual)==np.sign(anchor)).astype(float),.5)
    f['direction_advantage']=((np.sign(f.actual)==np.sign(f.prediction))&(f.prediction!=0)&(f.actual!=0)).astype(float)-anchor_hit
    # Collapse ALL contracts in a date before resampling time blocks.
    daily=f.groupby('target_date')[['improvement','direction_advantage','anchor_improvement']].mean().tail(policy.gate_days)
    if len(daily)<policy.gate_days: return {'passed':False,'reason':'insufficient_oos_dates','dates':len(daily)}
    vals=daily.to_numpy(); n=len(vals); rng=np.random.default_rng(20260911)
    blocks=int(np.ceil(n/policy.block_days))
    starts=rng.integers(0,n,size=(policy.bootstrap_samples,blocks))
    indices=(starts[:,:,None]+np.arange(policy.block_days))%n
    samples=vals[indices.reshape(policy.bootstrap_samples,-1)[:,:n]].mean(axis=1)
    lower=np.quantile(samples,policy.error_alpha/policy.family_tests,axis=0)
    stable=bool((vals[:n//2,[0,2]].mean(axis=0)>0).all() and (vals[n//2:,[0,2]].mean(axis=0)>0).all())
    passed=bool((lower>0).all() and stable)
    return {'passed':passed,'reason':'historical_screen_only' if passed else 'no_stable_advantage',
            'dates':n,'mae_improvement':float(vals[:,0].mean()),'mae_lower':float(lower[0]),
            'anchor_mae_improvement':float(vals[:,2].mean()),'anchor_mae_lower':float(lower[2]),
            'direction_lower':float(lower[1]),'stable_halves':stable,
            'unit':'date_cluster_circular_5day_block_bootstrap','forward_validated':False}

def conditional_radius(records,row,policy):
    known=[r for r in records if r['target_date']<=row['feature_date'] and r['sigma']>0]
    if not known: return None,{'reason':'no_calibration'}
    dates=sorted({r['target_date'] for r in known})[-policy.calibration_days:]
    known=[r for r in known if r['target_date'] in dates]
    scores={}
    for name,subset in [('contract',[r for r in known if r['symbol']==row['symbol']]),
                        ('sector',[r for r in known if r['sector']==row['sector']]),
                        ('family',[r for r in known if r['family']==row['family']]),
                        ('sector_regime',[r for r in known if r['sector']==row['sector'] and r['regime']==row['regime']])]:
        nd=len({r['target_date'] for r in subset})
        if nd<policy.min_calibration_dates: continue
        # Each date has equal mass; within-date equal contract weights.
        f=pd.DataFrame(subset); f['score']=abs(f.actual-f.prediction)/f.sigma
        f['w']=1/f.groupby('target_date').symbol.transform('count')
        f=f.sort_values('score'); c=f.w.cumsum()/f.w.sum()
        level=min(1.,policy.interval_mass*(1+1/nd))
        index=min(int(np.searchsorted(c,level)),len(f)-1)
        scores[name]={'q':float(f.iloc[index].score),'dates':nd}
    if not scores: return None,{'reason':'insufficient_calibration_dates'}
    radius=max(x['q'] for x in scores.values())*row['sigma']
    # OpenClaw is an unvalidated risk covariate; transparent conservative overlay,
    # not a learned return coefficient or an asserted coverage guarantee.
    risk=row.get('news_risk')
    multiplier=1.25 if risk is not None and np.isfinite(risk) and risk>=6 else 1.
    return float(radius*multiplier),{'strata':scores,'risk_multiplier':multiplier,
                                   'nominal_mass':policy.interval_mass,'guaranteed':False}

def replay(panel,asof,policy=StructuralPolicy(),candidates=('core','enriched'),start=None):
    panel=panel.loc[panel.feature_date<=asof].copy()
    family_size=2*len(candidates)*(1+panel.sector.nunique())
    if policy.family_tests<family_size:policy=replace(policy,family_tests=int(family_size))
    features=[c for c in panel if c.startswith(('local_','market_','sector_','life_','micro_','curve_','external_','news_'))]
    # Fixed universe dummies contain no outcome information.
    dummy=pd.get_dummies(panel[['sector','symbol']],dtype=float)
    results={}
    days=sorted(panel.feature_date.unique())
    for target in ['close','settle']:
      for candidate in candidates:
        cols=features if candidate=='enriched' else [c for c in features if not c.startswith(('curve_','external_','news_'))]
        matrix=panel[cols].to_numpy(float)
        missing=(~np.isfinite(matrix)).astype(float)
        x=np.column_stack([matrix,missing,dummy.to_numpy()])
        names=cols+[c+'_missing' for c in cols]+dummy.columns.tolist()
        y=panel['y_'+target].to_numpy(float); sigma=panel['sigma_'+target].to_numpy(float)
        anchor=panel.micro_close_vs_settle.fillna(0).to_numpy(float) if target=='settle' else np.zeros(len(panel))
        usable=np.isfinite(sigma)&(sigma>0)&panel.local_return_1d.notna().to_numpy()
        records=[]; latest=[]
        for day in days:
            if start and day<start: continue
            valid_dates=[d for d in days if d<day][-policy.max_train_days:]
            train=panel.feature_date.isin(valid_dates).to_numpy()&usable&np.isfinite(y)
            train &= panel.target_date.fillna('9999').le(day).to_numpy()
            query=panel.feature_date.eq(day).to_numpy()&usable
            ti=np.flatnonzero(train); qi=np.flatnonzero(query)
            if len(set(panel.iloc[ti].feature_date))<policy.min_train_days or len(qi)==0: continue
            tr=panel.iloc[ti]; counts=tr.groupby(['feature_date','sector']).symbol.transform('count').to_numpy()
            sec_counts=tr.groupby('feature_date').sector.transform('nunique').to_numpy()
            date_rank={d:i for i,d in enumerate(valid_dates)}
            age=np.array([len(valid_dates)-1-date_rank[d] for d in tr.feature_date])
            weights=2**(-age/policy.half_life_days)/counts/sec_counts
            # Forecast only the unknown increment beyond the known price anchor.
            # Settlement shrinkage must go toward today's last price, not toward
            # yesterday's volume-weighted settlement (a stale-price prior).
            pred,cont,intercept,fit=weighted_fit(x[ti],(y[ti]-anchor[ti])/sigma[ti],x[qi],weights,policy.ridge_alpha)
            trained=tr.groupby('symbol').size().to_dict()
            for j,idx in enumerate(qi):
                r=panel.iloc[idx]
                if trained.get(r.symbol,0)<policy.min_contract_rows: continue
                value=float(anchor[idx]+pred[j]*sigma[idx]*policy.shrinkage)
                groups={}
                for name,v in zip(names,cont[j]*sigma[idx]*policy.shrinkage):
                    key='hierarchy' if name.startswith('symbol_') or name.startswith('sector_') and name in dummy.columns else name.split('_')[0]
                    groups[key]=groups.get(key,0.)+float(v)
                groups['intercept']=float(intercept*sigma[idx]*policy.shrinkage)
                groups['known_price_anchor']=float(anchor[idx])
                regime='high' if (r.get('news_risk',np.nan)>=6 or r.local_daily_range>2*sigma[idx]) else 'normal'
                row={'symbol':r.symbol,'sector':r.sector,'family':r.family,'feature_date':day,
                     'target_date':r.target_date,'prediction':value,'actual':float(y[idx]) if np.isfinite(y[idx]) else None,
                     'sigma':float(sigma[idx]),'regime':regime,'news_risk':float(r.news_risk) if np.isfinite(r.news_risk) else None,
                     'training_dates':len(set(tr.feature_date)),'training_contract_rows':trained[r.symbol],
                     'training_rows':len(tr),'last_label_date':str(tr.target_date.max()),'contributions':groups}
                row['anchor_prediction']=float(r.micro_close_vs_settle) if target=='settle' and np.isfinite(r.micro_close_vs_settle) else 0.
                # Historical intervals are computed in one post-pass below to
                # avoid repeatedly scanning all accumulated rows while fitting.
                radius,calibration=(None,{'reason':'pending_postpass'})
                row.update(radius=radius,calibration=calibration)
                if day==asof:
                    row['features']={k:float(r[k]) if np.isfinite(r[k]) else None for k in cols}
                    row['fit']=fit; row['feature_names']=names
                    latest.append(row)
                if r.target_date is not None and np.isfinite(y[idx]): records.append(row)
        # Conditioning may only use labels realized by each forecast date.
        # Cache the shared date/sector/family quantiles; contract strata differ.
        for day in sorted({r['feature_date'] for r in records+latest}):
            current=[r for r in records+latest if r['feature_date']==day]
            known=[r for r in records if r['target_date']<=day]
            if not known: continue
            known_dates=sorted({r['target_date'] for r in known})[-policy.calibration_days:]
            known=[r for r in known if r['target_date'] in known_dates]
            df=pd.DataFrame(known);df['score']=abs(df.actual-df.prediction)/df.sigma
            cache={}
            def score(key,subset):
                if key in cache:return cache[key]
                nd=subset.target_date.nunique()
                if nd<policy.min_calibration_dates: cache[key]=None;return None
                sub=subset.copy();sub['w']=1/sub.groupby('target_date').symbol.transform('count');sub=sub.sort_values('score')
                cumulative=sub.w.cumsum()/sub.w.sum();level=min(1.,policy.interval_mass*(1+1/nd))
                idx=min(int(np.searchsorted(cumulative,level)),len(sub)-1)
                value={'q':float(sub.iloc[idx].score),'dates':int(nd)};cache[key]=value;return value
            for row in current:
                options={
                    'contract':score(('contract',row['symbol']),df.loc[df.symbol==row['symbol']]),
                    'sector':score(('sector',row['sector']),df.loc[df.sector==row['sector']]),
                    'family':score(('family',row['family']),df.loc[df.family==row['family']]),
                    'sector_regime':score(('sector_regime',row['sector'],row['regime']),df.loc[(df.sector==row['sector'])&(df.regime==row['regime'])])}
                options={k:v for k,v in options.items() if v is not None}
                if not options: row['calibration']={'reason':'insufficient_calibration_dates'};continue
                multiplier=1.25 if row.get('news_risk') is not None and row['news_risk']>=6 else 1.
                row['radius']=max(v['q'] for v in options.values())*row['sigma']*multiplier
                row['calibration']={'strata':options,'risk_multiplier':multiplier,'nominal_mass':policy.interval_mass,'guaranteed':False}
        gate=clustered_gate(records,policy,asof)
        results[target+'_'+candidate]={'records':records,'latest':latest,'gate':gate,
                                     'sector_gates':{s:clustered_gate([r for r in records if r['sector']==s],policy,asof) for s in panel.sector.unique()}}
    return {'asof':asof,'policy':asdict(policy),'forward_validated':False,'development_diagnostics':True,'models':results}

def night_reference(quote,close,settle,asof,cutoff=None):
    """Observed movement + zero remaining drift; no intraday alpha is invented."""
    if not quote or not quote.get('intraday_usable'): return None
    if not close or not settle or quote.get('price',0)<=0: return None
    timestamp=pd.Timestamp(quote['quote_date']+'T'+quote['quote_time']+'+08:00')
    # 23:00/01:00 night closes remain valid observations during their session
    # break. The large return is already observed, not a remaining-period alpha.
    beginning=pd.Timestamp(asof+'T21:00:00+08:00')
    if timestamp<beginning or timestamp>beginning+pd.Timedelta(hours=6): return None
    if cutoff and timestamp>pd.Timestamp(cutoff): return None
    if cutoff and (pd.Timestamp(cutoff)-timestamp).total_seconds()>12*3600: return None
    if quote.get('previous_settle') is None or not np.isclose(quote['previous_settle'],settle,rtol=1e-5): return None
    p=quote['price']
    return {'expected_close':p,'close_return':p/close-1,'close_vs_previous_settle':p/settle-1,
            'remaining_return':0.,'remaining_return_label':'martingale_reference_not_validated_intraday_alpha',
            'observed_quote_time':timestamp.isoformat(),'settlement_to_settlement_return':None,
            'can_trade':False,'forecast_type':'night_snapshot_reference'}
