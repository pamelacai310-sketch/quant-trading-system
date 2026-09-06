"""Project factor-engine forward paper experiment; separate from SMA control. No broker access.

Orders must precede quote time. Fill only available displayed volume, enforce
stock T+1, never backfill missed quotes, and maintain complete flat-to-flat PnL.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
from datetime import datetime,time
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.request import Request,urlopen
from zoneinfo import ZoneInfo
from .ledger import PositionLedger
from .forward_evidence import digest,write_once,make_orders,fetch_bars

TZ=ZoneInfo('Asia/Shanghai')
ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'experiments/project_paper_week_20260907'


def read(path):return json.loads(path.read_text())
def clock():return datetime.now(TZ)
def code_digest(config):
    return digest({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in config['source_paths']})


def parse_quotes(raw,config):
    result={}
    for item in config['instruments']:
        symbol=item['symbol'];key=(('sh' if symbol.startswith('6') else 'sz') if item['type']=='stock' else 'nf_')+symbol
        match=re.search(r'var hq_str_'+re.escape(key)+r'="([^"]*)";',raw)
        if not match:continue
        v=match[1].split(',')
        try:
            if item['type']=='stock':
                last,bid,ask,bsize,asize,volume=float(v[3]),float(v[11]),float(v[21]),float(v[10]),float(v[20]),float(v[8])
                stamp=v[30]+'T'+v[31]+'+08:00'
            else:
                last,bid,ask,bsize,asize,volume=float(v[3]),float(v[16]),float(v[26]),float(v[17]),float(v[27]),float(v[4])
                stamp=v[36]+'T'+v[37]+'+08:00'
            values=(last,bid,ask,bsize,asize,volume)
            if not all(math.isfinite(x) and x>=0 for x in values) or last<=0 or (bid and ask and bid>ask):
                raise ValueError('invalid market fields')
            datetime.fromisoformat(stamp)
        except (IndexError,ValueError):continue
        result[symbol]=dict(symbol=symbol,timestamp=stamp,last=last,bid=bid,ask=ask,bid_size=bsize,ask_size=asize,volume=volume,source='Sina displayed L1')
    return result


def fetch_quotes(config):
    keys=[(('sh' if i['symbol'].startswith('6') else 'sz') if i['type']=='stock' else 'nf_')+i['symbol'] for i in config['instruments']]
    req=Request('https://hq.sinajs.cn/list='+','.join(keys),headers={'Referer':'https://finance.sina.com.cn'})
    with urlopen(req,timeout=20) as response:raw=response.read().decode('gb18030')
    return raw,parse_quotes(raw,config)


def market_open(now):
    t=now.timetz().replace(tzinfo=None)
    return now.weekday()<5 and (time(9,30)<=t<time(11,30) or time(13)<=t<time(15))


def load_events(directory,config):
    lock=read(directory/'lock.json');previous=digest(lock);events=[]
    if lock['protocol_hash']!=digest(config) or lock['code_hash']!=code_digest(config):raise ValueError('Frozen source/protocol changed')
    for p in sorted((directory/'events').glob('*.json')):
        e=read(p)
        if e['previous_hash']!=previous or e['protocol_hash']!=digest(config):raise ValueError('Broken paper evidence chain')
        previous=digest(e);events.append(e)
    return events,previous


def freeze(directory=DIR):
    from .project_paper_model import prepare
    config=read(directory/'protocol.json')
    if (directory/'lock.json').exists():
        load_events(directory,config);return
    if clock()>=datetime.fromisoformat(config['start_at']):raise ValueError('Pre-registration deadline missed')
    config,bootstrap=prepare(config)
    from .project_paper_model import run_native
    bootstrap['native_preflight']=run_native({s:p for s,p in bootstrap['inputs'].items() if len(p.get('bars',[]))>=config['min_history']})
    if clock()>=datetime.fromisoformat(config['start_at']):raise ValueError('Preparation missed deadline')
    (directory/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    write_once(directory/'bootstrap.json',bootstrap)
    write_once(directory/'lock.json',dict(frozen_at=clock().isoformat(),protocol_hash=digest(config),code_hash=code_digest(config),bootstrap_hash=digest(bootstrap)))


def initial_state(config,bootstrap):
    states={s:{i['symbol']:asdict(PositionLedger(i.get('initial_cash',config['cash_per_sleeve']),i['multiplier'],i['type']=='futures')) for i in config['instruments']} for s in ('candidate','baseline')}
    return dict(states=states,orders={},marks={},cycles={},closed_trades=[],bought_today={},last_fill_quote={},
                last_model_date=config['bootstrap_date'],model_inputs=bootstrap['inputs'],peak_equity={s:sum(i.get('initial_cash',config['cash_per_sleeve']) for i in config['instruments']) for s in states},halted=False)


def queue_orders(config,state,now):
    from .project_paper_model import model_targets
    targets,audit=model_targets(config,state['model_inputs'],state['states'],state['last_model_date'])
    state['selection_audit']=audit
    orders={}
    for strategy,items in targets.items():
        orders[strategy]={}
        for symbol,t in items.items():
            order={**t,'created_at':clock().isoformat(),'strategy':strategy,'symbol':symbol,'status':'pending'}
            order['order_id']=digest(order)
            if t['target_quantity']==state['states'][strategy][symbol]['quantity']:order['status']='no_change'
            orders[strategy][symbol]=order
    state['orders']=orders


def execute_quotes(config,state,quotes,now):
    fills=[];rejects=[]
    for item in config['instruments']:
        symbol=item['symbol']
        if symbol not in quotes:
            rejects.append(dict(symbol=symbol,reason='missing_or_malformed_quote'));continue
        quote=quotes[symbol];ts=datetime.fromisoformat(quote['timestamp'])
        age=(now-ts).total_seconds()
        if age<0 or age>config['quote_max_age_seconds'] or ts.date()!=now.date():
            rejects.append({'symbol':symbol,'reason':'stale_or_future_quote'});continue
        state['marks'][symbol]={'price':quote['last'],'quote_time':quote['timestamp']}
        for strategy in state['states']:
            order=state['orders'].get(strategy,{}).get(symbol)
            if not order or order['status'] not in ('pending','partial'):continue
            if now>=datetime.fromisoformat(config['end_at']):continue
            created=datetime.fromisoformat(order['created_at'])
            if ts<=created or state['last_fill_quote'].get(strategy+':'+symbol)==quote['timestamp']:continue
            if (now-created).total_seconds()>config['order_ttl_seconds']:
                order['status']='expired';continue
            ledger=PositionLedger(**state['states'][strategy][symbol]);delta=order['target_quantity']-ledger.quantity
            if not delta:order['status']='filled';continue
            spread=(quote['ask']-quote['bid'])/quote['last']*10000
            if abs(order['target_quantity'])>abs(ledger.quantity) and (quote['bid']<=0 or quote['ask']<=0 or spread>config['max_spread_bps']):
                rejects.append(dict(symbol=symbol,strategy=strategy,reason='spread_or_one_sided_book'));continue
            size=quote['ask_size'] if delta>0 else quote['bid_size']
            best=quote['ask'] if delta>0 else quote['bid']
            if best<=0 or size<item['lot']:continue
            amount=min(abs(delta),math.floor(size*config['book_participation']/item['lot'])*item['lot'])
            # Stock exits can sell only previously settled shares.
            today_key=f'{strategy}:{symbol}:{now.date()}'
            if item['type']=='stock' and delta<0:
                amount=min(amount,max(0,ledger.quantity-state['bought_today'].get(today_key,0)))
            amount=math.floor(amount/item['lot'])*item['lot']
            if amount<=0:continue
            signed=math.copysign(amount,delta)
            raw=best*(1+math.copysign(item['slippage_bps']/10000,delta))
            price=(math.ceil(raw/item['tick']) if delta>0 else math.floor(raw/item['tick']))*item['tick']
            notional=amount*price*item['multiplier']
            fee=notional*(item['fee_bps']+(item['sell_tax_bps'] if delta<0 else 0))/10000
            if item['type']=='stock':fee=max(notional*item['fee_bps']/10000,5.)+(notional*item['sell_tax_bps']/10000 if delta<0 else 0)
            if abs(ledger.quantity+signed)>abs(ledger.quantity) and (2*fee/notional*10000+2*item['slippage_bps']+2*max(0,spread)+item['sell_tax_bps'])>config['max_roundtrip_cost_bps']:
                rejects.append(dict(symbol=symbol,strategy=strategy,reason='small_fill_cost_budget'));continue
            hypothetical=PositionLedger(**asdict(ledger));old=ledger.quantity
            if abs(old+signed)>abs(old) and abs(old+signed)*price*item['multiplier']>max(0,ledger.equity(quote['last']))*config['notional_fraction']:
                rejects.append(dict(symbol=symbol,strategy=strategy,reason='notional_risk_cap'));continue
            realized=hypothetical.fill(signed,price,fee)
            equity=hypothetical.equity(quote['last'])
            if equity<=0 or (item['type']=='stock' and hypothetical.cash<0) or abs(hypothetical.quantity)*quote['last']*item['multiplier']*item['margin_rate']>equity:
                rejects.append(dict(symbol=symbol,strategy=strategy,reason='capital_or_margin'));continue
            key=strategy+':'+symbol
            if old==0:
                state['cycles'][key]=dict(symbol=symbol,strategy=strategy,entry_time=quote['timestamp'],net_pnl=0.,entry_notional=0.)
            cycle=state['cycles'][key]
            if realized is None:cycle['entry_notional']+=notional
            else:cycle['net_pnl']+=realized['net_pnl']
            state['states'][strategy][symbol]=asdict(hypothetical)
            if signed>0 and item['type']=='stock':state['bought_today'][today_key]=state['bought_today'].get(today_key,0)+signed
            if hypothetical.quantity==0:
                state['closed_trades'].append({**cycle,'exit_time':quote['timestamp']})
                del state['cycles'][key]
            order['status']='filled' if hypothetical.quantity==order['target_quantity'] else 'partial'
            state['last_fill_quote'][key]=quote['timestamp']
            fills.append(dict(order_id=order['order_id'],symbol=symbol,strategy=strategy,observed_at=now.isoformat(),quote_time=quote['timestamp'],quantity=signed,price=price,fee=fee,book_price=best,realized=realized,mode='paper'))
    return fills,rejects


def step(directory=DIR):
    config=read(directory/'protocol.json');now=clock()
    start,end=datetime.fromisoformat(config['start_at']),datetime.fromisoformat(config['end_at'])
    events,previous=load_events(directory,config)
    if now<start:return {'status':'armed_waiting_for_start'}
    if events and events[-1]['kind']=='finalized':return {'status':'already_finalized'}
    bootstrap=read(directory/'bootstrap.json')
    if digest(bootstrap)!=read(directory/'lock.json')['bootstrap_hash']:raise ValueError('Bootstrap tampered')
    state=copy.deepcopy(events[-1]['state']) if events else initial_state(config,bootstrap)
    if 'model_evidence_hash' in state:
        model_evidence=read(directory/'decisions'/(state['model_evidence_hash']+'.json'))
        if digest(model_evidence)!=state['model_evidence_hash']:raise ValueError('Model evidence tampered')
        state.update(model_evidence)
    raw=None;quotes={};fills=[];rejects=[];kind='checkpoint'
    if now>=end:
        for items in state['orders'].values():
            for order in items.values():
                if order['status'] in ('pending','partial'):order['status']='expired_at_cutoff'
        kind='finalized'
    elif not events:
        queue_orders(config,state,now);kind='started'
    elif market_open(now):
        # Do not open new Friday positions that cannot complete stock T+1.
        if now.date().isoformat()==config['sessions'][-1]:
            for orders in state['orders'].values():
                for order in orders.values():
                    if order['target_quantity']!=0:
                        order.update(target_quantity=0,created_at=now.isoformat(),status='pending',reason='final_session_exit')
                        order['order_id']=digest(order)
        raw,quotes=fetch_quotes(config)
        received=clock()
        if received<end and market_open(received):fills,rejects=execute_quotes(config,state,quotes,received)
        now=received
    elif now.hour>=15 and now.date().isoformat() in config['sessions'] and state['last_model_date']!=str(now.date()):
        from .project_paper_model import collect
        data=collect(config['instruments'],str(now.date()))
        state['model_inputs']=data;state['last_model_date']=str(now.date())
        # Closing marks are valuations only, never retrospective fills.
        for symbol,payload in data.items():
            if 'bars' not in payload:continue
            state['marks'][symbol]={'price':payload['bars'][-1]['close'],'quote_time':str(now.date())+'T15:00:00+08:00','source':'daily_close_valuation_only'}
        queue_orders(config,state,clock());kind='next_session_orders'
    else:return {'status':'market_closed_no_fill'}
    # Drawdown protection cancels risk increases and queues exits for next fresh quote.
    initial=sum(i.get('initial_cash',config['cash_per_sleeve']) for i in config['instruments'])
    for strategy in state['states']:
        nav=sum(PositionLedger(**v).equity(state['marks'].get(s,{'price':v['average_price']})['price']) for s,v in state['states'][strategy].items())
        peak=max(state['peak_equity'][strategy],nav);state['peak_equity'][strategy]=peak
        if nav/peak-1 < -config['max_drawdown']:state['halted']=True
    if state['halted'] and now<end:
        for strategy,items in state['orders'].items():
            for symbol,order in items.items():
                if order.get('reason')!='drawdown_exit':
                    order.update(target_quantity=0,created_at=now.isoformat(),status='pending',reason='drawdown_exit')
                    order['order_id']=digest(order)
    model_evidence={key:state.pop(key) for key in ('model_inputs','selection_audit') if key in state}
    model_hash=digest(model_evidence);model_path=directory/'decisions'/(model_hash+'.json')
    if not model_path.exists():write_once(model_path,model_evidence)
    elif digest(read(model_path))!=model_hash:raise ValueError('Model evidence collision')
    state['model_evidence_hash']=model_hash
    event=dict(kind=kind,recorded_at=clock().isoformat(),protocol_hash=digest(config),previous_hash=previous,raw_quote_response=raw,quotes=quotes,fills=fills,rejections=rejects,state=state)
    write_once(directory/'events'/f'{len(events):06d}.json',event)
    return {'status':kind,'fills':len(fills),'event_hash':digest(event)}


def report(directory=DIR):
    config=read(directory/'protocol.json');events,_=load_events(directory,config)
    state=events[-1]['state'] if events else initial_state(config,read(directory/'bootstrap.json'))
    if 'model_evidence_hash' in state:
        evidence=read(directory/'decisions'/(state['model_evidence_hash']+'.json'))
        if digest(evidence)!=state['model_evidence_hash']:raise ValueError('Model evidence tampered')
        state['selection_audit']=evidence.get('selection_audit')
    initial=sum(i.get('initial_cash',config['cash_per_sleeve']) for i in config['instruments']);result={}
    for strategy,items in state['states'].items():
        trades=[t for t in state['closed_trades'] if t['strategy']==strategy];pnl=[t['net_pnl'] for t in trades]
        wins=[x for x in pnl if x>0];losses=[x for x in pnl if x<0];positions=[];nav=0
        for symbol,v in items.items():
            mark=state['marks'].get(symbol,{'price':v['average_price'],'quote_time':None})
            nav+=PositionLedger(**v).equity(mark['price'])
            if v['quantity']:positions.append({'symbol':symbol,**v,'mark':mark})
        gross=sum(abs(v['quantity'])*state['marks'].get(symbol,{'price':v['average_price']})['price']*v['multiplier'] for symbol,v in items.items())
        paid_cost=sum(f['fee']+abs(f['quantity'])*(abs(f['price']-f['book_price']))*items[f['symbol']]['multiplier'] for e in events for f in e['fills'] if f['strategy']==strategy)
        result[strategy]=dict(gross_exposure=gross/initial,net_profit_at_double_recorded_cost=nav-initial-paid_cost,starting_cash=initial,ending_nav=nav,net_profit=nav-initial,net_return=nav/initial-1,
                             closed_trade_count=len(trades),win_rate=len(wins)/len(trades) if trades else None,
                             payoff_ratio=(sum(wins)/len(wins))/(-sum(losses)/len(losses)) if wins and losses else None,
                             realized_closed_trade_net_profit=sum(pnl),open_positions=positions,
                             fill_count=sum(f['strategy']==strategy for e in events for f in e['fills']))
    output=dict(experiment=config['experiment_id'],as_of=clock().isoformat(),start_at=config['start_at'],end_at=config['end_at'],
                status='finalized' if events and events[-1]['kind']=='finalized' else 'in_progress' if events else 'not_started',
                actual_start_at=next((e['recorded_at'] for e in events if e['kind']=='started'),None),
                observed_quote_days=sorted({q['timestamp'][:10] for e in events for q in e['quotes'].values()}),
                pending_orders=state['orders'],
                model=config['model'],accounts=result,events=len(events),data_failures=len(list((directory/'failures').glob('*.json'))),
                selection_audit=state.get('selection_audit'),repeatable_advantage_proven=False,notes=['Net NAV includes unrealized PnL; open positions are not closed wins.','Payoff ratio is mean winning net PnL / absolute mean losing net PnL; null if either group is empty.','Display-book simulated fills and declared costs; not actual broker execution.'])
    write_path=directory/'summary.json';write_path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','step','report','preflight']);args=parser.parse_args()
    try:
        if args.action=='freeze':freeze();print('frozen');return
        if args.action=='preflight':
            config=read(DIR/'protocol.json');raw,quotes=fetch_quotes(config)
            write_once(DIR/'preflight.json',dict(received_at=clock().isoformat(),quotes=quotes,raw=raw,note='Connectivity/schema check only. Weekend quotes are not executable.'))
            print('quote connectivity/schema passed; no fills')
        elif args.action=='step':print(json.dumps(step(),indent=2))
        else:print(json.dumps(report(),indent=2))
    except Exception as exc:
        write_once(DIR/'failures'/f'{clock().strftime("%Y%m%dT%H%M%S%f")}.json',dict(at=clock().isoformat(),error=str(exc),status='blocked_no_backfill'))
        raise

if __name__=='__main__':main()
