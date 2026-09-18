"""Chronological additive return forecasts with shared component risk gates.

Equal sector weights prevent sectors with more contracts dominating the market.
These are statistical components, not identified causal effects. Missing peers
are not imputed; a fixed, complete panel is required for comparable components.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HierarchyPolicy:
    min_train: int = 40
    lookback: int = 120
    validation_days: int = 60
    min_validation: int = 30
    ridge: float = 10.0
    block_days: int = 5

    def __post_init__(self):
        if (self.min_train < 10 or self.lookback < self.min_train or
                self.min_validation < 10 or self.validation_days < self.min_validation
                or self.block_days < 2 or not np.isfinite(self.ridge) or self.ridge <= 0):
            raise ValueError('Invalid hierarchy policy')


def decompose(returns, sectors):
    if returns.empty or returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError('Unique ordered dates required')
    if returns.columns.has_duplicates or set(returns.columns) != set(sectors):
        raise ValueError('Exact fixed universe and sector mapping required')
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError('Missing returns: supply an explicitly selected complete panel')
    groups = {s: [c for c in returns if sectors[c] == s] for s in sorted(set(sectors.values()))}
    sector_means = pd.DataFrame({s: returns[c].mean(axis=1) for s,c in groups.items()})
    market = sector_means.mean(axis=1)
    sector = sector_means.sub(market, axis=0)
    residual = pd.DataFrame({c: returns[c]-sector_means[sectors[c]] for c in returns})
    return market, sector, residual


def _forecast(series, policy):
    # Labels are next-session component returns. Query uses only the last row.
    frame = pd.DataFrame({'lag1': series, 'mean5': series.rolling(5).mean(),
                          'mean20': series.rolling(20).mean(), 'target': series.shift(-1)})
    train=frame.iloc[:-1].dropna().tail(policy.lookback)
    if len(train)<policy.min_train or frame.iloc[-1,:3].isna().any(): return None
    x=train.iloc[:,:3].to_numpy(); y=train.target.to_numpy()
    mean=x.mean(axis=0); sd=x.std(axis=0); sd=np.where(sd>1e-12,sd,1.)
    x=(x-mean)/sd; q=(frame.iloc[-1,:3].to_numpy()-mean)/sd
    beta=np.linalg.solve(x.T@x+policy.ridge*np.eye(3), x.T@(y-y.mean()))
    return float(y.mean()+q@beta)


def _weight(history, policy):
    h=history[-policy.validation_days:]
    if len(h)<policy.min_validation:
        return 0., {'status':'insufficient_oos', 'dates':len(h)}
    gain=np.array([abs(r['actual'])-abs(r['actual']-r['prediction']) for r in h])
    blocks=[gain[i:i+policy.block_days].mean() for i in range(0,len(gain),policy.block_days)
            if len(gain[i:i+policy.block_days])==policy.block_days]
    se=float(np.std(blocks,ddof=1)/np.sqrt(len(blocks))) if len(blocks)>1 else float('inf')
    # Conservative fixed heuristic, not a formal multiple-testing significance test.
    lower=float(gain.mean()-2*se)
    stable=min(gain[:len(gain)//2].mean(),gain[len(gain)//2:].mean())>0
    weight=1. if lower>0 and stable else 0.
    return weight, {'status':'historical_support' if weight else 'no_stable_advantage',
                   'dates':len(h),'mae_gain':float(gain.mean()),'block_lower_diagnostic':lower,
                   'stable_halves':bool(stable),'forward_validated':False}


def replay_hierarchy(returns, sectors, asof, policy=HierarchyPolicy()):
    panel=returns.loc[returns.index<=asof].copy()
    market, industry, residual=decompose(panel,sectors)
    series={'market':market}
    series.update({'sector:'+s:industry[s] for s in industry})
    series.update({'specific:'+c:residual[c] for c in residual})
    history={k:[] for k in series}; pending={}; records=[]; latest=[]
    for i,day in enumerate(panel.index):
        for key,forecast in pending.items():
            history[key].append({'target_date':str(day),'prediction':forecast,'actual':float(series[key].iloc[i])})
        predictions={k:_forecast(v.iloc[:i+1],policy) for k,v in series.items()}
        predictions={k:v for k,v in predictions.items() if v is not None}
        weights={k:_weight(history[k],policy) for k in predictions}
        day_rows=[]
        for symbol in panel:
            keys=['market','sector:'+sectors[symbol],'specific:'+symbol]
            if not all(k in predictions for k in keys): continue
            components={k:predictions[k] for k in keys}
            scaled={k:predictions[k]*weights[k][0] for k in keys}
            row={'symbol':symbol,'feature_date':str(day),'components':components,
                 'risk_weights':{k:weights[k][0] for k in keys},
                 'validation':{k:weights[k][1] for k in keys},
                 'raw_prediction':sum(components.values()),'risk_scaled_prediction':sum(scaled.values()),
                 'status':'RESEARCH_ONLY','forward_validated':False}
            if i+1<len(panel):
                row.update(target_date=str(panel.index[i+1]),actual=float(panel[symbol].iloc[i+1]))
                records.append(row)
            day_rows.append(row)
        latest=day_rows
        pending=predictions
    return {'asof':str(asof),'latest':latest,'records':records,'component_oos':history,
            'production_enabled':False,'definition':'market + sector excess + within-sector residual'}
