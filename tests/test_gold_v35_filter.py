"""Synthetic *unit tests*, never interpreted as historical performance."""
import unittest
import pandas as pd
from quant_trade_system.gold_v35_filter import backtest, point_in_time_merge, latest_p0_status

def market():
    days=pd.bdate_range('2025-01-06',periods=11)
    gc=[2700,2680,2670,2660,2665,2682,2690,2694,2700,2710,2715]
    return pd.DataFrame({
        'date':days,
        'cutoff_at':(days+pd.Timedelta(hours=23)).astype(str),
        'next_au_open_at':(days+pd.Timedelta(days=1,hours=3)).astype(str),
        'gc_close':gc,
        'gc_high':[x+10 for x in gc],
        'gc_low':[x-10 for x in gc],
        'tips10_pp':[2,1.97,1.94,1.91,1.89,1.88,1.85,1.84,1.80,1.76,1.74],
        'dxy':[106-i*.1 for i in range(len(days))],
        'au_next_open':[600.]*len(days),
        'au_next_close':[602.]*len(days)
    })

def etf():
    return pd.DataFrame({'available_at':['2025-01-06T12:00:00Z','2025-01-14T12:00:00Z'],
                         'etf_tonnes':[900,903]})

def cot():
    return pd.DataFrame({'available_at':['2025-01-03T20:30:00Z','2025-01-10T20:30:00Z'],
                         'cot_mm_net':[100000,101000]})

class GoldP0Test(unittest.TestCase):
    def test_trigger_and_price_flow_release(self):
        frame,report=backtest(market(),etf(),cot())
        self.assertTrue(frame.loc[1,'divergence'])
        self.assertTrue(frame.loc[2,'p0_block_new_long'])
        self.assertGreater(report['blocked_baseline_longs'],0)
        self.assertTrue(frame['p0_release_confirmed'].any())
        self.assertTrue(frame.loc[frame['p0_release_confirmed'],'selling_pressure_exhausted'].all())
        self.assertLess(report['v35']['trades'],report['baseline']['trades'])
    def test_missing_flows_fail_closed(self):
        frame,_=backtest(market(),pd.DataFrame(),pd.DataFrame())
        self.assertTrue(frame.loc[1:,'p0_block_new_long'].all())
        self.assertFalse(frame['p0_release_confirmed'].any())
    def test_cftc_tuesday_is_not_available_until_friday_release(self):
        source=pd.DataFrame({'available_at':['2025-01-10T20:30:00Z'],'cot_mm_net':[80000]})
        merged=point_in_time_merge(market().iloc[:6],source,'cot_mm_net','cot_mm_net')
        self.assertTrue(pd.isna(merged.iloc[2]['cot_mm_net']))
        self.assertEqual(merged.iloc[4]['cot_mm_net'],80000)
    def test_cannot_trade_before_signal_availability(self):
        m=market()
        m.loc[2,'next_au_open_at']='2025-01-08T12:00:00Z'
        with self.assertRaises(ValueError):
            backtest(m,etf(),cot())
    def test_live_guard_does_not_need_future_prices(self):
        history=market().drop(columns=['next_au_open_at','au_next_open','au_next_close'])
        result=latest_p0_status(history.iloc[:4],etf(),cot())
        self.assertTrue(result['p0_block_new_long'])
        self.assertIn('not a buy signal',result['policy'])

if __name__=='__main__':
    unittest.main()
