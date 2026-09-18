"""Export a structural replay through the research evidence overlay.

Costs must be supplied per contract as decimal round-trip returns. Missing
costs remain missing. This CLI does not change the nightly production route.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

module_path=Path(__file__).resolve().parents[1]/'quant_trade_system/core/direction_evidence.py'
spec=importlib.util.spec_from_file_location('direction_evidence_export',module_path)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--costs')
    args=parser.parse_args()
    replay=json.loads(Path(args.input).read_text())
    costs=json.loads(Path(args.costs).read_text()) if args.costs else {}
    model=replay['models']['close_enriched']
    rows=[module.assess_direction(r,model['records'],cost=costs.get(r['symbol']),
          forward_validated=replay.get('forward_validated') is True,
          gate_passed=model['gate'].get('passed') is True) for r in model['latest']]
    with Path(args.output).open('x',encoding='utf-8') as f:
        json.dump({'asof':replay['asof'],'decisions':rows,
                   'costs_supplied':bool(args.costs),'production_enabled':False},
                  f,ensure_ascii=False,indent=2,allow_nan=False)

if __name__=='__main__':main()
