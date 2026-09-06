import json
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime
from unittest.mock import patch
import unittest
from quant_trade_system import forward_evidence as f


class ForwardTests(unittest.TestCase):
    def setup_dir(self,path):
        config=json.loads((f.DEFAULT/'protocol.json').read_text())
        (path/'protocol.json').write_text(json.dumps(config))
        return config

    def bars(self,item,date):
        return {'source':'TEST_FIXTURE_ONLY','bars':[{'date':date,'open':100.,'high':102.,'low':99.,'close':101.,'volume':1e9} for _ in range(20)]}

    def test_freeze_replay_idempotence_and_tamper(self):
        with TemporaryDirectory() as tmp,patch.object(f,'fetch_bars',self.bars):
            p=Path(tmp);self.setup_dir(p)
            f.run(p,datetime(2026,9,6,18,tzinfo=f.TZ))
            f.run(p,datetime(2026,9,7,18,tzinfo=f.TZ))
            self.assertEqual(f.run(p,datetime(2026,9,7,19,tzinfo=f.TZ))['status'],'already_captured')
            report=f.summarize(p)
            self.assertFalse(report['repeatable_advantage_proven'])
            self.assertEqual(report['captured_days'],1)
            self.assertEqual(report['trade_statistics']['candidate']['closed_trade_count'],0)
            first=p/'observations/2026-09-04.json'
            record=json.loads(first.read_text());record['orders']['baseline']['600519']['target_quantity']+=100
            first.write_text(json.dumps(record))
            with self.assertRaises(ValueError): f.summarize(p)

    def test_no_late_bootstrap_or_parameter_change(self):
        with TemporaryDirectory() as tmp,patch.object(f,'fetch_bars',self.bars):
            p=Path(tmp);config=self.setup_dir(p)
            with self.assertRaises(ValueError):f.run(p,datetime(2026,9,7,10,tzinfo=f.TZ))
            f.run(p,datetime(2026,9,6,18,tzinfo=f.TZ))
            config['lookback']=5;(p/'protocol.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):f.run(p,datetime(2026,9,7,18,tzinfo=f.TZ))

    def test_data_failure_does_not_write_fake_observations(self):
        with TemporaryDirectory() as tmp,patch.object(f,'fetch_bars',side_effect=ValueError('missing')):
            p=Path(tmp);self.setup_dir(p)
            with self.assertRaises(ValueError):f.run(p,datetime(2026,9,6,18,tzinfo=f.TZ))
            self.assertFalse((p/'observations').exists())

    def test_flat_roundtrip_shared_accounting(self):
        with TemporaryDirectory() as tmp,patch.object(f,'fetch_bars',self.bars):
            p=Path(tmp);config=self.setup_dir(p)
            f.run(p,datetime(2026,9,6,18,tzinfo=f.TZ))
            prev=json.loads((p/'observations/2026-09-04.json').read_text())
            data={i['symbol']:self.bars(i,'2026-09-07') for i in config['instruments']}
            state,fills,eq,res=f.replay(config,prev,data,'2026-09-07')
            prev['states']=state
            for strategy in prev['orders']:
                for order in prev['orders'][strategy].values():order['target_quantity']=0
            state,closed,eq,res=f.replay(config,prev,data,'2026-09-08')
            for strategy in ('candidate','baseline'):
                pnl=sum(t['realized']['net_pnl'] for t in closed if t['strategy']==strategy)
                self.assertAlmostEqual(eq[strategy]-40000000,pnl)
                self.assertLess(pnl,0)

if __name__=='__main__':unittest.main()
