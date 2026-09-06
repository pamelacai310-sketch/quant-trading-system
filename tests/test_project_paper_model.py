import copy
from dataclasses import asdict
from datetime import datetime
import unittest
from quant_trade_system import project_paper_week as p
from quant_trade_system.project_paper_model import model_targets, run_native
from quant_trade_system.ledger import PositionLedger

class ProjectPaperTests(unittest.TestCase):
    def setUp(self):
        self.c=p.read(p.DIR/'protocol.json')
        self.c['instruments']=[dict(symbol='600519',type='stock',multiplier=1,lot=100,fee_bps=3,sell_tax_bps=5,slippage_bps=5,tick=.01,margin_rate=1)]
        self.data={'600519':{'bars':[dict(date='2026-09-04',open=100,high=101,low=99,close=100,volume=10000000) for _ in range(100)]}}
        self.states={s:{'600519':asdict(PositionLedger(10000000))} for s in ('candidate','baseline')}
    def native(self,data):
        return dict(status='trained',symbols={'600519':{'status':'trained','latest_signal_score':.8}},trade_actions=[dict(symbol='600519',action='LONG',target_weight=.2)])
    def test_orders_require_native_action_and_audit_every_symbol(self):
        orders,a=model_targets(self.c,self.data,self.states,'2026-09-04',self.native)
        self.assertGreater(orders['candidate']['600519']['target_quantity'],0)
        self.assertTrue(a['candidates'][0]['selected'])
        orders,a=model_targets(self.c,self.data,self.states,'2026-09-04',lambda d:dict(status='no_actionable_signals'))
        self.assertEqual(orders['candidate']['600519']['target_quantity'],0)
        self.assertIn('no_native_trade_action',a['candidates'][0]['reasons'])
    def test_stale_input_and_model_error_fail_closed(self):
        self.data['600519']['bars'][-1]['date']='2026-09-03'
        orders,a=model_targets(self.c,self.data,self.states,'2026-09-04',self.native)
        self.assertEqual(orders['candidate']['600519']['target_quantity'],0)
        self.assertIn('missing_or_stale_data',a['candidates'][0]['reasons'])
        def fail(d):raise RuntimeError('model unavailable')
        orders,a=model_targets(self.c,self.data,self.states,'2026-09-04',fail)
        self.assertEqual(a['native_output']['status'],'model_error')
    def test_missing_quote_and_wide_spread_cannot_open(self):
        state=p.initial_state(self.c,dict(inputs=self.data))
        state['orders']={s:{'600519':dict(target_quantity=100,created_at='2026-09-07T09:00:00+08:00',status='pending',order_id=s)} for s in self.states}
        now=datetime(2026,9,7,9,35,tzinfo=p.TZ)
        self.assertEqual(p.execute_quotes(self.c,state,{},now)[0],[])
        q={'600519':dict(timestamp=now.isoformat(),last=100,bid=99,ask=101,bid_size=10000,ask_size=10000)}
        fills,rejects=p.execute_quotes(self.c,state,q,now)
        self.assertFalse(fills);self.assertEqual(rejects[0]['reason'],'spread_or_one_sided_book')
    def test_concurrent_position_cap_blocks_new_fill(self):
        c=copy.deepcopy(self.c);c['max_positions']=0
        state=p.initial_state(c,dict(inputs=self.data))
        state['orders']={s:{'600519':dict(target_quantity=100,created_at='2026-09-07T09:00:00+08:00',status='pending',order_id=s)} for s in self.states}
        now=datetime(2026,9,7,9,35,tzinfo=p.TZ)
        quotes={'600519':dict(timestamp=now.isoformat(),last=100,bid=100,ask=100,bid_size=10000,ask_size=10000)}
        fills,rejects=p.execute_quotes(c,state,quotes,now)
        self.assertFalse(fills);self.assertEqual(rejects[0]['reason'],'concurrent_position_or_family_cap')

    def test_decision_evidence_reused_and_tampering_blocks(self):
        import json
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        from quant_trade_system.forward_evidence import digest
        c=copy.deepcopy(self.c);c['source_paths']=[]
        bootstrap={'inputs':self.data}
        with TemporaryDirectory() as td:
            d=Path(td)
            for name,value in [('protocol',c),('bootstrap',bootstrap),('lock',dict(protocol_hash=digest(c),code_hash=p.code_digest(c),bootstrap_hash=digest(bootstrap)))]:
                (d/(name+'.json')).write_text(json.dumps(value))
            with patch.object(p,'clock',return_value=datetime(2026,9,7,9,tzinfo=p.TZ)), patch('quant_trade_system.project_paper_model.run_native',self.native):
                self.assertEqual(p.step(d)['status'],'started')
            with patch.object(p,'clock',return_value=datetime(2026,9,7,9,35,tzinfo=p.TZ)),patch.object(p,'fetch_quotes',return_value=('',{})):
                p.step(d)
            self.assertEqual(len(list((d/'decisions').glob('*.json'))),1)
            decision=next((d/'decisions').glob('*.json'));decision.write_text('{}')
            with self.assertRaises(ValueError):p.report(d)

    def test_native_entrypoint_executes_without_substituted_model(self):
        result=run_native({})
        self.assertEqual(result['status'],'no_data')

if __name__=='__main__':unittest.main()
