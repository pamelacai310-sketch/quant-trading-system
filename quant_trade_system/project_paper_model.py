"""Predeclared universe and native project-engine adapter. No fallback signals."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
import numpy as np
import pandas as pd
from .ledger import PositionLedger
from .forward_evidence import digest


def collect(instruments, cutoff):
    import akshare as ak
    def one(item):
        try:
            symbol=item['symbol']
            if item['type']=='stock':
                frame=ak.stock_zh_a_daily(symbol=('sh' if symbol.startswith('6') else 'sz')+symbol,
                    start_date='20250101',end_date=cutoff.replace('-',''),adjust='')
            else:frame=ak.futures_zh_daily_sina(symbol=symbol)
            frame=frame[['date','open','high','low','close','volume']].copy()
            frame['date']=pd.to_datetime(frame['date']).dt.strftime('%Y-%m-%d')
            frame=frame[frame.date<=cutoff].sort_values('date').tail(252)
            if frame.empty or frame.date.iloc[-1]!=cutoff:raise ValueError('stale daily data')
            if frame.date.duplicated().any():raise ValueError('duplicate dates')
            vals=frame[['open','high','low','close','volume']].astype(float)
            if not np.isfinite(vals.to_numpy()).all() or (vals<=0).any().any():raise ValueError('nonpositive or missing data')
            if ((vals.high<vals[['open','close','low']].max(axis=1)) | (vals.low>vals[['open','close']].min(axis=1))).any():raise ValueError('invalid OHLC')
            return symbol,{'source':'AKShare '+ak.__version__,'bars':json.loads(frame.to_json(orient='records'))}
        except Exception as exc:return item['symbol'],{'error':str(exc),'cutoff':cutoff}
    with ThreadPoolExecutor(max_workers=6) as pool:return dict(pool.map(one,instruments))


def prepare(config):
    from pathlib import Path
    snapshot=json.loads((Path(__file__).resolve().parents[1]/config['universe_snapshot_path']).read_text())
    records=snapshot['symbols'];codes=sorted(set(records))
    if len(codes)!=50:raise ValueError('Expected complete frozen SSE50 snapshot')
    def item(symbol,kind,mult=1):
        return dict(symbol=symbol,type=kind,multiplier=mult,lot=200 if symbol.startswith('688') else 100 if kind=='stock' else 1,
            fee_bps=3 if kind=='stock' else 1,sell_tax_bps=5 if kind=='stock' else 0,
            slippage_bps=5,tick=.01 if kind=='stock' else .2,margin_rate=1 if kind=='stock' else .2)
    instruments=[item(s,'stock') for s in codes]
    instruments += [item(f'{family}{month}','futures',300 if family in ('IF','IH') else 200)
                    for family in ('IF','IH','IC','IM') for month in ('2609','2612')]
    for i in instruments:i['initial_cash']=config['initial_capital']/2/(50 if i['type']=='stock' else 8)
    config={**config,'instruments':instruments}
    data=collect(instruments,config['bootstrap_date'])
    bootstrap=dict(inputs=data,universe_source=snapshot['source'],universe_snapshot=records,
                   captured_at=datetime.now().astimezone().isoformat(),universe_rule=config['universe_rule'])
    return config,bootstrap


def run_native(data):
    from .core.causal.self_iterating_causal_engine import SelfIteratingCausalEngine
    frames={s:pd.DataFrame(p['bars']).assign(date=lambda f:pd.to_datetime(f.date)).set_index('date') for s,p in data.items() if 'bars' in p}
    return SelfIteratingCausalEngine().run_learning_cycle(frames)


def model_targets(config,data,states,cutoff,runner=None):
    audit={};valid={}
    for item in config['instruments']:
        s=item['symbol'];payload=data.get(s,{})
        row={'symbol':s,'input_hash':digest(payload),'data_cutoff':cutoff,'reasons':[]}
        bars=payload.get('bars',[])
        if not bars or bars[-1]['date']!=cutoff:row['reasons'].append('missing_or_stale_data')
        elif len(bars)<config['min_history']:row['reasons'].append('insufficient_history')
        else:
            vals=np.array([[b[k] for k in ('open','high','low','close','volume')] for b in bars],dtype=float)
            if not np.isfinite(vals).all() or (vals<=0).any():row['reasons'].append('invalid_data')
            else:valid[s]=payload
            row['adv20_notional']=float(np.mean([b['close']*b['volume']*item['multiplier'] for b in bars[-20:]]))
        audit[s]=row
    try: native=(runner or run_native)(valid)
    except Exception as exc:native={'status':'model_error','error':str(exc),'symbols':{},'trade_actions':[]}
    actions={a['symbol']:a for a in native.get('trade_actions',[]) if a.get('action') in ('LONG','SHORT') and a.get('symbol') in audit}
    eligible=[]
    for item in config['instruments']:
        s=item['symbol'];row=audit[s];report=native.get('symbols',{}).get(s,{})
        row.update(model_status=report.get('status',native.get('status')),score=report.get('latest_signal_score'),confidence=report.get('latest_confidence'),native_action=actions.get(s))
        if not row['reasons'] and row.get('adv20_notional',0)<config['min_adv_notional']:row['reasons'].append('liquidity_floor')
        roundtrip=2*(item['fee_bps']+item['slippage_bps'])+item['sell_tax_bps']+2*config['max_spread_bps']
        row['roundtrip_cost_budget_bps']=roundtrip
        if roundtrip>config['max_roundtrip_cost_bps']:row['reasons'].append('cost_budget')
        if not row['reasons']:eligible.append(s)
    # At most one contract per index family, chosen by observed ADV, not fixed December preference.
    for family in ('IF','IH','IC','IM'):
        contracts=sorted([s for s in eligible if s.startswith(family)],key=lambda s:(-audit[s]['adv20_notional'],s))
        for s in contracts[1:]:eligible.remove(s);audit[s]['reasons'].append('less_liquid_same_index_contract')
    baseline_set=set(sorted(eligible,key=lambda s:(-audit[s]['adv20_notional'],s))[:config['max_positions']])
    ranked=sorted([s for s in eligible if s in actions],key=lambda s:(-abs(audit[s]['score'] or 0),s))
    chosen=set(ranked[:config['max_positions']])
    result={strategy:{} for strategy in states}
    total=sum(i.get('initial_cash',config['cash_per_sleeve']) for i in config['instruments'])
    for item in config['instruments']:
        s=item['symbol'];row=audit[s];bars=data.get(s,{}).get('bars',[])
        price=bars[-1]['close'] if bars else 0
        if s not in actions:row['reasons'].append('no_native_trade_action')
        elif s in ranked and s not in chosen:row['reasons'].append('position_count_cap')
        row['selected']=s in chosen
        for strategy in states:
            ledger=PositionLedger(**states[strategy][s]);direction=0;weight=0
            if strategy=='candidate' and s in chosen:
                direction=1 if actions[s]['action']=='LONG' else -1
                weight=max(0,float(actions[s].get('target_weight',0)))
            elif strategy=='baseline' and s in baseline_set:
                mean=sum(b['close'] for b in bars[-20:])/20
                direction=1 if price>mean else -1 if item['type']=='futures' else 0
                weight=1
            if item['type']=='stock' and direction<0:
                direction=0
                if strategy=='candidate':row['reasons'].append('stock_short_not_supported');row['selected']=False
            cap=min(max(0,ledger.equity(price))*config['notional_fraction'],total*weight,row.get('adv20_notional',0)*config['max_participation'])
            qty=direction*math.floor(cap/(price*item['multiplier']*item['lot']*1.02))*item['lot'] if price>0 else 0
            # Preserve units while direction survives; reverse only after a separate flat fill.
            if ledger.quantity:qty=ledger.quantity if qty*ledger.quantity>0 else 0
            if not qty and direction and strategy=='candidate':row['reasons'].append('below_one_lot_or_zero_weight');row['selected']=False
            result[strategy][s]=dict(target_quantity=qty,reference_close=price,known_at=cutoff+'T15:00:00+08:00',decision='native_project_engine' if strategy=='candidate' else 'liquidity_top5_sma20_control')
    return result,dict(cutoff=cutoff,inputs_hash=digest(data),native_output=native,candidates=list(audit.values()),
        note='Native score is not calibrated expected return. Cost budget is an execution filter, not proof of positive expected value.')
