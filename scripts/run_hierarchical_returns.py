"""Run the additive research challenger on explicit complete return panels.

JSON input: asof, returns [{date, SYMBOL: decimal_return}], sectors {SYMBOL: sector}.
No automatic missing-value fill, production activation, or broker orders.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pandas as pd

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    path=Path(__file__).resolve().parents[1]/'quant_trade_system/core/hierarchical_returns.py'
    spec=importlib.util.spec_from_file_location('hierarchical_returns_cli',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    raw=Path(args.input).read_bytes();obj=json.loads(raw)
    frame=pd.DataFrame(obj['returns']).set_index('date');frame.index=pd.to_datetime(frame.index)
    result=module.replay_hierarchy(frame,obj['sectors'],pd.Timestamp(obj['asof']))
    result['input_sha256']=hashlib.sha256(raw).hexdigest()
    result['source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with Path(args.output).open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)

if __name__=='__main__':main()
