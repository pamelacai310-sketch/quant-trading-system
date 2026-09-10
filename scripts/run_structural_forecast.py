"""Replay exact-contract close/settlement research with immutable output.

Input JSON keys: calendar, asof, contracts {symbol: bar rows}, and optional
auxiliary {dated/vendor-main symbol: rows}, external_events {series: events},
news [{available_at: timezone-aware ISO time, risk, ...}].
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quant_trade_system.structural_forecast import build_panel,replay

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',default=None)
    args=p.parse_args();path=Path(args.input);obj=json.loads(path.read_text())
    panel,_=build_panel({s:pd.DataFrame(v) for s,v in obj['contracts'].items()},obj['calendar'],obj['asof'],
                        obj.get('auxiliary'),obj.get('external_events'),obj.get('news'))
    result=replay(panel,obj['asof'],start=args.start)
    result['input_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    result['trade_actions']=[]
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)

if __name__=='__main__':main()
