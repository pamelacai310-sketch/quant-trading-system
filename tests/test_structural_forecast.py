import unittest
import numpy as np
import pandas as pd
from quant_trade_system.structural_forecast import (
    StructuralPolicy, build_panel, replay, clustered_gate, point_in_time,
    night_reference, weighted_fit,
)

def bars(n=115):
    rng=np.random.default_rng(37)
    c=100*np.cumprod(1+rng.normal(0,.01,n))
    return pd.DataFrame({'date':pd.bdate_range('2026-01-02',periods=n).strftime('%Y-%m-%d'),
        'close':c,'open':c*.998,'high':c*1.01,'low':c*.99,'settle':c*(1+rng.normal(0,.003,n)),
        'volume':np.arange(n)*50+500,'hold':np.arange(n)*100+1000})

class StructuralTests(unittest.TestCase):
    def test_news_released_after_decision_cannot_enter_features(self):
        events=[{'available_at':'2026-09-09T08:05:00+08:00','risk':4},
                {'available_at':'2026-09-10T20:00:00+08:00','risk':9}]
        actual=point_in_time(events,['2026-09-10','2026-09-11'],'risk')
        self.assertEqual(actual.tolist(),[4,9])
        with self.assertRaises(ValueError): point_in_time([{'available_at':'2026-09-10','risk':4}],['2026-09-10'],'risk')

    def test_future_prices_labels_and_news_do_not_change_frozen_prediction(self):
        f=bars(); dates=f.date.tolist(); day=dates[95]
        panel,_=build_panel({'CU2612':f,'AL2612':f.copy()},dates,day)
        policy=StructuralPolicy(min_train_days=30,gate_days=10,min_calibration_dates=5,bootstrap_samples=30)
        first=replay(panel,day,policy,candidates=('core',),start=dates[85])
        later=f.copy();later.loc[96:,['open','high','low','close','settle']]*=10
        changed,_=build_panel({'CU2612':later,'AL2612':later.copy()},dates,day,
            news=[{'available_at':dates[96]+'T08:00:00+08:00','risk':10}])
        second=replay(changed,day,policy,candidates=('core',),start=dates[85])
        self.assertEqual(first,second)
        for m in first['models'].values():
            for r in m['latest']: self.assertLessEqual(r['last_label_date'],day)

    def test_target_definitions_do_not_substitute_last_trade_for_settlement(self):
        f=bars(35); dates=f.date.tolist()
        panel,_=build_panel({'CU2612':f},dates,dates[-1])
        row=panel.loc[panel.feature_date==dates[-2]].iloc[0]
        self.assertAlmostEqual(row.y_close,f.close.iloc[-1]/f.close.iloc[-2]-1)
        self.assertAlmostEqual(row.y_settle,f.settle.iloc[-1]/f.settle.iloc[-2]-1)
        self.assertNotAlmostEqual(row.y_close,row.y_settle)
        f.loc[33,'volume']=0
        panel,_=build_panel({'CU2612':f},dates,dates[-1])
        self.assertTrue(np.isnan(panel.loc[panel.feature_date==dates[32],'y_settle'].iloc[0]))

    def test_more_correlated_contract_copies_do_not_increase_gate_evidence(self):
        p=StructuralPolicy(gate_days=60,bootstrap_samples=100)
        rows=[{'target_date':str(d.date()),'prediction':.002,'actual':.003 if i%3 else -.004}
              for i,d in enumerate(pd.bdate_range('2026-01-02',periods=60))]
        self.assertEqual(clustered_gate(rows,p),clustered_gate(rows*57,p))

    def test_fit_ignores_missing_values_and_contributions_reconcile(self):
        x=np.array([[1,np.nan],[2,np.nan],[3,np.nan]],float)
        p,c,b,fit=weighted_fit(x,np.array([.1,.2,.3]),np.array([[4,np.nan]]),np.ones(3),30)
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(float(p[0]),float(c[0].sum()+b))

    def test_settlement_gate_must_beat_known_close_anchor(self):
        p=StructuralPolicy(gate_days=60,bootstrap_samples=100)
        rows=[{'target_date':str(d.date()),'prediction':.008,'actual':.01,'anchor_prediction':.01}
              for d in pd.bdate_range('2026-01-02',periods=60)]
        gate=clustered_gate(rows,p)
        self.assertGreater(gate['mae_improvement'],0)
        self.assertLess(gate['anchor_mae_lower'],0)
        self.assertFalse(gate['passed'])

    def test_settlement_shrinkage_preserves_already_known_anchor(self):
        f=bars(95);f['settle']=f.close.shift(1)
        dates=f.date.tolist();panel,_=build_panel({'CU2612':f},dates,dates[-1])
        p=StructuralPolicy(min_train_days=30,min_contract_rows=20,bootstrap_samples=20)
        result=replay(panel,dates[-1],p,candidates=('core',),start=dates[-1])
        r=result['models']['settle_core']['latest'][0]
        self.assertAlmostEqual(r['prediction'],f.close.iloc[-1]/f.settle.iloc[-1]-1,places=10)

    def test_midnight_session_reference_and_previous_settlement_guard(self):
        q={'price':98.,'quote_date':'2026-09-11','quote_time':'01:00:00','previous_settle':100.,'intraday_usable':True}
        r=night_reference(q,101.,100.,'2026-09-10','2026-09-11T01:14:00+08:00')
        self.assertAlmostEqual(r['close_return'],98/101-1)
        self.assertIsNone(r['settlement_to_settlement_return'])
        self.assertFalse(r['can_trade'])
        self.assertIsNone(night_reference(q,101.,103.,'2026-09-10'))
        self.assertIsNone(night_reference(q,101.,100.,'2026-09-10','2026-09-10T22:00:00+08:00'))

if __name__=='__main__': unittest.main()
