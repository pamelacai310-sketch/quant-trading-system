import importlib.util
import pathlib
import sys
import unittest

path=pathlib.Path(__file__).parents[1]/'quant_trade_system/core/direction_evidence.py'
spec=importlib.util.spec_from_file_location('direction_evidence_tested',path)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

class DirectionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(symbol='X',feature_date='2026-09-17',prediction=.00001,sigma=.02)
    def test_tiny_edge_neutral(self):
        self.assertEqual(m.assess_direction(self.row,[],cost=.001)['direction'],'NEUTRAL')
    def test_missing_cost_never_admits(self):
        self.row['prediction']=.01
        self.assertEqual(m.assess_direction(self.row,[],forward_validated=True,gate_passed=True)['action'],'OBSERVE_ONLY')
    def test_future_outcomes_do_not_change_result(self):
        future=dict(symbol='X',feature_date='2026-09-17',target_date='2026-09-18',prediction=.01,actual=.5)
        self.assertEqual(m.assess_direction(self.row,[]),m.assess_direction(self.row,[future]))
    def test_other_contract_ignored(self):
        r=dict(symbol='Y',feature_date='2026-09-15',target_date='2026-09-16',prediction=.01,actual=.5)
        self.assertEqual(m.assess_direction(self.row,[r])['sample_dates'],0)
    def test_invalid_sigma(self):
        self.row['sigma']=float('nan')
        self.assertIn('invalid_prediction',m.assess_direction(self.row,[])['reasons'])
    def test_unvalidated_large_edge_observed(self):
        self.row['prediction']=-.03
        r=m.assess_direction(self.row,[],cost=.001)
        self.assertEqual(r['direction'],'SHORT');self.assertEqual(r['action'],'OBSERVE_ONLY')
    def test_duplicate_outcomes_rejected(self):
        r=dict(symbol='X',feature_date='2026-09-15',target_date='2026-09-16',prediction=.01,actual=.5)
        with self.assertRaises(ValueError):m.assess_direction(self.row,[r,r])
    def test_negative_cost_not_admitted(self):
        r=m.assess_direction(self.row,[],cost=-1)
        self.assertIn('execution_cost_unavailable',r['reasons'])

if __name__=='__main__':unittest.main()
