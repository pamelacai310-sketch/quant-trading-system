"""Frozen, daily shadow experiment. Never connects to a trading broker.

Each successful run archives inputs, next-session orders and full account states.
An unavailable observation is a failed run, never a synthetic zero return.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from .ledger import PositionLedger

TZ = ZoneInfo('Asia/Shanghai')
ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'experiments/p0_week_20260907'


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_once(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')


def code_hash():
    paths = sorted((ROOT/'quant_trade_system').rglob('*.py'))
    return digest({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def fetch_bars(instrument, date):
    import akshare as ak
    symbol=instrument['symbol']
    if instrument['type']=='stock':
        frame=ak.stock_zh_a_daily(symbol=('sh' if symbol.startswith('6') else 'sz')+symbol,start_date='20260601',end_date=date.replace('-',''),adjust='')
        # Sina daily stock volume is shares, not lots. No corporate-action adjustment.
        source='akshare.stock_zh_a_daily:Sina:unadjusted'
    else:
        if not symbol[-4:].isdigit():
            raise ValueError('Only specific dated futures contracts accepted')
        frame=ak.futures_zh_daily_sina(symbol=symbol)
        source='akshare.futures_zh_daily_sina:specific_contract'
    frame=frame[['date','open','high','low','close','volume']].copy()
    frame['date']=pd.to_datetime(frame['date']).dt.strftime('%Y-%m-%d')
    frame=frame.loc[frame.date <= date].sort_values('date').tail(60)
    numeric=['open','high','low','close','volume']
    frame[numeric]=frame[numeric].apply(pd.to_numeric,errors='raise')
    if len(frame)<20 or frame.date.iloc[-1]!=date or frame.date.duplicated().any():
        raise ValueError(f'{symbol}: stale, duplicate or insufficient data for {date}')
    if not np.isfinite(frame[numeric]).all().all() or (frame[numeric]<=0).any().any():
        raise ValueError(f'{symbol}: invalid prices or volume')
    if ((frame.high<frame[['open','close','low']].max(axis=1)) | (frame.low>frame[['open','close','high']].min(axis=1))).any():
        raise ValueError(f'{symbol}: invalid OHLC')
    return {'source':source,'provider_version':ak.__version__,'bars':frame.to_dict('records')}


def make_orders(config, data, state):
    orders={}
    for strategy in ('candidate','baseline'):
        orders[strategy]={}
        for item in config['instruments']:
            symbol=item['symbol'];bars=data[symbol]['bars'];last=bars[-1]
            ledger=PositionLedger(**state[strategy][symbol])
            sign=1
            if strategy=='candidate' and last['close'] <= np.mean([b['close'] for b in bars[-config['lookback']:]]):
                sign=-1 if item['type']=='futures' else 0
            # Exits flatten first; subsequent session may enter the opposite direction.
            target=ledger.quantity
            if ledger.quantity and np.sign(ledger.quantity)!=sign:
                target=0
            elif not ledger.quantity and sign:
                budget=max(0,ledger.equity(last['close']))*config['notional_fraction']
                target=sign*math.floor(budget/(last['close']*item['multiplier']*item['lot']*1.01))*item['lot']
            orders[strategy][symbol]={'target_quantity':target,'reference_close':last['close'],
                                      'known_at':last['date'],'decision':'sma20' if strategy=='candidate' else 'long'}
    return orders


def replay(config, previous, data, date):
    states=json.loads(json.dumps(previous['states']))
    fills=[];equities={};stress_equities={}
    for strategy in ('candidate','baseline'):
        equities[strategy]=0;stress_equities[strategy]=0
        for item in config['instruments']:
            symbol=item['symbol'];bar=data[symbol]['bars'][-1]
            order=previous['orders'][strategy][symbol]
            ledger=PositionLedger(**states[strategy][symbol])
            old_qty=ledger.quantity
            delta=order['target_quantity']-old_qty
            if bar['high']==bar['low']:
                raise ValueError(f'{symbol}: single-price bar; cannot assert fill')
            if abs(bar['open']/order['reference_close']-1)>.095:
                raise ValueError(f'{symbol}: limit/corporate-action/gap requires manual review')
            if abs(delta)>config['max_participation']*bar['volume']:
                raise ValueError(f'{symbol}: shadow volume cap exceeded')
            if delta:
                # Round adverse slippage to tick; costs are declared assumptions.
                raw=bar['open']*(1+np.sign(delta)*item['slippage_bps']/10000)
                price=(math.ceil(raw/item['tick']) if delta>0 else math.floor(raw/item['tick']))*item['tick']
                notional=abs(delta)*price*item['multiplier']
                fee=notional*(item['fee_bps']+(item['sell_tax_bps'] if delta<0 else 0))/10000
                if item['type']=='stock': fee=max(fee,5.0)
                realized=ledger.fill(delta,price,fee)
                fills.append({'strategy':strategy,'symbol':symbol,'date':date,'quantity':delta,
                              'price':price,'fee':fee,'slippage_cost':abs(price-bar['open'])*abs(delta)*item['multiplier'],
                              'realized':realized,'fill_quality':'daily_bar_estimate_not_actual_execution'})
            mark=ledger.equity(bar['close'])
            margin=abs(ledger.quantity)*bar['close']*item['multiplier']*item['margin_rate']
            if mark<=0 or margin>mark:
                raise ValueError(f'{symbol}: margin/capital breach')
            states[strategy][symbol]=asdict(ledger)
            equities[strategy]+=mark
            # Reserve closing cost on open positions; Friday is not a fabricated close.
            exit_notional=abs(ledger.quantity)*bar['close']*item['multiplier']
            exit_cost=exit_notional*(item['fee_bps']+item['slippage_bps']+(item['sell_tax_bps'] if ledger.quantity>0 else 0))/10000
            if ledger.quantity and item['type']=='stock': exit_cost=max(exit_cost,5)
            stress_equities[strategy]+=mark-exit_cost
    return states,fills,equities,stress_equities


def run(directory=DEFAULT, now=None):
    clock_injected = now is not None
    now=now or datetime.now(TZ)
    config=json.loads((directory/'protocol.json').read_text())
    lock_path=directory/'lock.json'
    sessions=config['sessions']
    if now.strftime('%Y-%m-%d')>sessions[-1]:
        return {'status':'window_closed_no_backfill'}
    if not lock_path.exists():
        if now >= datetime.fromisoformat(sessions[0]+'T09:30:00').replace(tzinfo=TZ):
            raise ValueError('Missed freeze deadline: cannot manufacture pre-registration')
        write_once(lock_path,{'created_at':now.isoformat(),'protocol_hash':digest(config),'code_hash':code_hash()})
    lock=json.loads(lock_path.read_text())
    if lock['protocol_hash']!=digest(config) or lock['code_hash']!=code_hash():
        raise ValueError('Frozen protocol or source code changed')
    date=now.strftime('%Y-%m-%d')
    bootstrap=date<sessions[0]
    if bootstrap: date=config['bootstrap_date']
    elif date not in sessions or now.hour<16:
        return {'status':'outside_capture_window'}
    path=directory/'observations'/f'{date}.json'
    if path.exists():
        return {'status':'already_captured','date':date}
    prev_date=config['bootstrap_date'] if date==sessions[0] else (sessions[sessions.index(date)-1] if not bootstrap else None)
    previous=None
    if prev_date:
        previous=json.loads((directory/'observations'/f'{prev_date}.json').read_text())
        prior_time=datetime.fromisoformat(previous['captured_at'])
        if prior_time>=datetime.fromisoformat(date+'T09:30:00').replace(tzinfo=TZ):
            raise ValueError('Signal was not committed before session open')
        if previous['next_session']!=date:
            raise ValueError('Incorrect next-session commitment')
    data={item['symbol']:fetch_bars(item,date) for item in config['instruments']}
    if bootstrap:
        states={s:{i['symbol']:asdict(PositionLedger(config['cash_per_sleeve'],i['multiplier'],i['type']=='futures')) for i in config['instruments']} for s in ('candidate','baseline')}
        fills=[];equities={s:config['cash_per_sleeve']*len(config['instruments']) for s in states};reserved=dict(equities)
    else:
        states,fills,equities,reserved=replay(config,previous,data,date)
    stop=any(equities[s]/max(equities[s],(previous or {}).get('peak_equity',{}).get(s,equities[s]))-1 < -config['max_drawdown'] for s in equities)
    orders=make_orders(config,data,states)
    if stop:
        raise ValueError('Drawdown stop: invalidate run and require reconciliation; no further orders')
    next_session=sessions[0] if bootstrap else (sessions[sessions.index(date)+1] if date!=sessions[-1] else None)
    completed = now if clock_injected else datetime.now(TZ)
    if bootstrap and completed >= datetime.fromisoformat(sessions[0]+'T09:30:00').replace(tzinfo=TZ):
        raise ValueError('Bootstrap fetch missed pre-open commitment deadline')
    output={'date':date,'captured_at':completed.isoformat(),'protocol_hash':digest(config),'code_hash':code_hash(),
            'previous_hash':digest(previous) if previous else digest(lock),'next_session':next_session,
            'inputs':data,'states':states,'fills':fills,'equity':equities,'liquidation_reserved_equity':reserved,
            'peak_equity':{s:max(equities[s],(previous or {}).get('peak_equity',{}).get(s,equities[s])) for s in equities},
            'orders':orders if next_session else {},'status':'bootstrap' if bootstrap else 'preliminary_shadow_evidence',
            'repeatable_advantage_proven':False,'production_eligible':False}
    write_once(path,output)
    return {'status':output['status'],'date':date,'observation_hash':digest(output)}


def summarize(directory=DEFAULT):
    config=json.loads((directory/'protocol.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted((directory/'observations').glob('*.json'))]
    initial=config['cash_per_sleeve']*len(config['instruments'])
    lock_path=directory/'lock.json'
    previous=json.loads(lock_path.read_text()) if lock_path.exists() else None
    for r in records:
        if previous is None or r['previous_hash']!=digest(previous) or r['protocol_hash']!=digest(config):
            raise ValueError('Broken evidence chain or modified protocol')
        previous=r
    rows=[r for r in records if r['date'] in config['sessions']]
    net={s:rows[-1]['liquidation_reserved_equity'][s]/initial-1 if rows else None for s in ('candidate','baseline')}
    fills=[f for row in rows for f in row['fills']]
    stats={}
    stress={}
    for strategy in ('candidate','baseline'):
        selected=[f for f in fills if f['strategy']==strategy]
        closed=[f['realized']['net_pnl'] for f in selected if f['realized'] is not None]
        costs=sum(f['fee']+f['slippage_cost'] for f in selected)
        # Fixed-order stress: double already-incurred costs, no re-optimization.
        stress[strategy]=net[strategy]-costs/initial if rows else None
        daily=[initial]+[r['equity'][strategy] for r in rows]
        a=np.asarray(daily,float)
        stats[strategy]={'fill_count':len(selected),'closed_trade_count':len(closed),
                         'win_rate':float(np.mean(np.asarray(closed)>0)) if closed else None,
                         'net_closed_pnl':sum(closed),'costs':costs,
                         'payoff_ratio':float(np.mean([x for x in closed if x>0])/-np.mean([x for x in closed if x<0])) if any(x>0 for x in closed) and any(x<0 for x in closed) else None,
                         'max_drawdown':float(np.min(a/np.maximum.accumulate(a)-1))}
    return {'status':'insufficient_evidence','trade_statistics':stats,'double_incurred_cost_return':stress,'captured_days':len(rows),'required_days':len(config['sessions']),
            'net_marked_return_with_exit_reserve':net,
            'excess_return':net['candidate']-net['baseline'] if rows else None,
            'positive_absolute_return':bool(rows and net['candidate']>0),
            'costs_verified':config['costs_verified'],'repeatable_advantage_proven':False,
            'limitations':['five days below predeclared independent-day minimum','bar-estimated fills, not broker executions','cost assumptions not verified','no intraday queue or gap-stop guarantee'],
            'frozen_parameters':config}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['capture','report']);parser.add_argument('--directory',type=Path,default=DEFAULT)
    args=parser.parse_args()
    if args.action=='report':
        print(json.dumps(summarize(args.directory),indent=2));return
    try:
        print(json.dumps(run(args.directory),indent=2))
    except Exception as exc:
        stamp=datetime.now(TZ)
        error={'status':'blocked_no_synthetic_fallback','captured_at':stamp.isoformat(),'error':str(exc),'repeatable_advantage_proven':False}
        write_once(args.directory/'failures'/f'{stamp.strftime("%Y%m%dT%H%M%S%f")}.json',error)
        print(json.dumps(error,indent=2));raise SystemExit(1)

if __name__=='__main__': main()
