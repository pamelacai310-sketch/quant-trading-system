import copy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
from quant_trade_system import paper_week as p
from quant_trade_system.ledger import PositionLedger


class PaperWeekTests(unittest.TestCase):
    def setUp(self):
        self.c=p.read(p.DIR/'protocol.json')
        self.b=p.read(p.ROOT/'experiments/p0_week_20260907/observations/2026-09-04.json')
        self.state=p.initial_state(self.c,self.b)
        self.start=datetime(2026,9,7,9,tzinfo=p.TZ)
        p.queue_orders(self.c,self.state,self.start)

    def quotes(self,stamp,price=100):
        return {i['symbol']:dict(symbol=i['symbol'],timestamp=stamp,last=price,bid=price,ask=price,bid_size=100,ask_size=100,volume=1000000) for i in self.c['instruments']}

    def target(self,amount):
        for strategy in self.state['orders']:
            for symbol,order in self.state['orders'][strategy].items():
                order.update(target_quantity=amount if symbol=='600519' else 0,status='pending')

    def test_next_fresh_quote_and_duplicate_protection(self):
        self.target(200)
        now=datetime(2026,9,7,9,35,tzinfo=p.TZ)
        quotes=self.quotes('2026-09-07T09:35:00+08:00')
        fills,_=p.execute_quotes(self.c,self.state,quotes,now)
        self.assertEqual(len(fills),2)
        self.assertEqual(self.state['orders']['candidate']['600519']['status'],'partial')
        self.assertEqual(p.execute_quotes(self.c,self.state,quotes,now)[0],[])
        self.assertEqual(self.state['states']['candidate']['600519']['quantity'],100)

    def test_stale_future_and_preorder_quotes_never_fill(self):
        self.target(100);now=datetime(2026,9,7,9,35,tzinfo=p.TZ)
        for stamp in ('2026-09-04T15:00:00+08:00','2026-09-07T09:36:00+08:00'):
            self.assertEqual(p.execute_quotes(self.c,self.state,self.quotes(stamp),now)[0],[])
        for orders in self.state['orders'].values():
            for order in orders.values():order['created_at']=now.isoformat()
        self.assertEqual(p.execute_quotes(self.c,self.state,self.quotes(now.isoformat()),now)[0],[])

    def test_t_plus_one_and_complete_cycle_net_pnl(self):
        self.target(100);now=datetime(2026,9,7,9,35,tzinfo=p.TZ)
        p.execute_quotes(self.c,self.state,self.quotes(now.isoformat()),now)
        self.target(0);later=datetime(2026,9,7,10,tzinfo=p.TZ)
        self.assertEqual(p.execute_quotes(self.c,self.state,self.quotes(later.isoformat()),later)[0],[])
        later=datetime(2026,9,8,9,35,tzinfo=p.TZ)
        fills,_=p.execute_quotes(self.c,self.state,self.quotes(later.isoformat()),later)
        self.assertEqual(len(fills),2)
        self.assertEqual(len(self.state['closed_trades']),2)
        for t in self.state['closed_trades']:
            self.assertLess(t['net_pnl'],0)
            ledger=PositionLedger(**self.state['states'][t['strategy']]['600519'])
            self.assertAlmostEqual(ledger.cash-self.c['cash_per_sleeve'],t['net_pnl'])

    def test_cutoff_prevents_fills(self):
        self.target(100);now=datetime(2026,9,12,2,tzinfo=p.TZ)
        self.assertEqual(p.execute_quotes(self.c,self.state,self.quotes(now.isoformat()),now)[0],[])

    def test_order_lifecycle_no_afterhours_fills(self):
        with TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'protocol.json').write_text(json.dumps(self.c))
            with patch.object(p,'clock',return_value=datetime(2026,9,6,12,tzinfo=p.TZ)):p.freeze(d)
            with patch.object(p,'clock',return_value=self.start):self.assertEqual(p.step(d)['status'],'started')
            with patch.object(p,'clock',return_value=datetime(2026,9,7,12,tzinfo=p.TZ)):
                self.assertEqual(p.step(d)['status'],'market_closed_no_fill')
            with patch.object(p,'clock',return_value=datetime(2026,9,12,2,tzinfo=p.TZ)):
                self.assertEqual(p.step(d)['status'],'finalized')
                r=p.report(d)
            self.assertEqual(r['accounts']['candidate']['closed_trade_count'],0)
            self.assertIsNone(r['accounts']['candidate']['win_rate'])
            self.assertEqual(r['accounts']['candidate']['net_profit'],0)
            self.assertFalse(r['repeatable_advantage_proven'])

if __name__=='__main__':unittest.main()
