import math
import unittest
import numpy as np
import pandas as pd
from quant_trade_system.ledger import PositionLedger, position_returns
from quant_trade_system.backtest import backtest_strategy
from quant_trade_system.core.formulas import evaluate_formula
from quant_trade_system.core.robustness import deflated_sharpe_ratio, evaluate_cpcv_returns
from quant_trade_system.core.statistical_learning_layer import StatisticalLearningLayer, PYTORCH_AVAILABLE


class P0Tests(unittest.TestCase):
    def test_short_profit_fees_and_partial_cost_basis(self):
        for futures in (False, True):
            ledger = PositionLedger(10000, multiplier=10, futures=futures)
            ledger.fill(-2, 100, 2)
            ledger.fill(-2, 120, 2)
            self.assertEqual(ledger.average_price, 110)
            trade = ledger.fill(1, 90, 1)
            self.assertEqual(trade['net_pnl'], 198)
            self.assertEqual(ledger.average_price, 110)
            ledger.fill(3, 90, 3)
            self.assertAlmostEqual(ledger.equity(90), 10792)

    def test_bearish_correct_never_turns_long_loss_into_profit(self):
        layer = StatisticalLearningLayer()
        out = layer._compute_metrics(np.array([0,0]), np.array([.1,.9]), np.array([-.02,-.02]), cost_bps=0)
        self.assertLess(out['net_total_return'], 0)
        self.assertIsNone(out['win_rate'])
        self.assertAlmostEqual(out['direction_accuracy'], .5)
        np.testing.assert_allclose(position_returns([-1,1], [-.02,-.02]), [.02,-.02])

    def test_dsr_frequency_invariance_and_known_probability(self):
        x = np.tile([-1.,1.],126)
        r = pd.Series(.01*(x + .5/math.sqrt(252)*x.std(ddof=1)))
        a = deflated_sharpe_ratio(r,periods_per_year=252)
        b = deflated_sharpe_ratio(r,periods_per_year=12)
        self.assertEqual(a['dsr_probability'],b['dsr_probability'])
        self.assertGreater(a['dsr_probability'], .68)
        self.assertLess(a['dsr_probability'], .71)
        self.assertFalse(evaluate_cpcv_returns(r)['passed'])
        self.assertFalse(deflated_sharpe_ratio(r,effective_trials=20)['passed_dsr_95'])

    def test_formula_long_lag_underscores_and_unsafe_syntax(self):
        x=pd.Series(np.arange(1.,26.))
        r=evaluate_formula('(money_supply_t - money_supply_t-12) / money_supply_t-12',{'money_supply':x})
        self.assertAlmostEqual(r.iloc[-1],12/13)
        r=evaluate_formula('policy_rate_t - policy_rate_t-1',{'policy_rate':x})
        self.assertEqual(r.iloc[-1],1)
        r=evaluate_formula('std(policy_rate_t-4:t) / mean(policy_rate_t-4:t)',{'policy_rate':x})
        self.assertAlmostEqual(r.iloc[-1],x.iloc[-4:].std()/x.iloc[-4:].mean())
        for bad in ("__import__('os')", 'missing / money_supply', 'money_supply.shift(-1)'):
            with self.assertRaises(ValueError): evaluate_formula(bad,{'money_supply':x})

    def run_bars(self, rows, fee=0):
        frame=pd.DataFrame(rows,columns=['open','high','low','close'])
        frame['timestamp']=pd.date_range('2026-01-01',periods=len(frame))
        frame['volume']=10000
        spec={'indicators':[], 'entry_rules':[{'left':'close','op':'>','right':100}],
              'exit_rules':[{'left':'close','op':'<','right':100}],
              'position_sizing':{'risk_fraction':.5,'max_units':1}}
        return backtest_strategy('p0','p0',frame,spec,starting_cash=1000,fee_bps=fee,slippage_bps=0)

    def test_no_same_bar_fill_exit_path_or_unrealized_win(self):
        result=self.run_bars([(101,200,1,101),(120,125,115,99),(90,1000,1,90)])
        self.assertEqual(result.trades[0]['price'],120)
        self.assertEqual(result.trades[0]['timestamp'],pd.Timestamp('2026-01-02'))
        self.assertAlmostEqual(result.total_return,-.03)
        trade=result.stats['closed_trades'][0]
        self.assertEqual(trade['net_pnl'],-30)
        self.assertAlmostEqual(trade['mfe_pct'],100*5/120,places=5)
        self.assertAlmostEqual(result.max_drawdown,-.03)
        open_result=self.run_bars([(101,101,101,101),(120,130,115,130)])
        self.assertEqual(open_result.win_rate,0)
        self.assertEqual(open_result.stats['closed_trades'],[])
        self.assertIsNotNone(open_result.stats['open_position'])

    def test_flat_price_is_losing_after_round_trip_costs(self):
        result=self.run_bars([(101,101,101,101),(100,101,99,99),(100,100,100,100)],fee=10)
        self.assertLess(result.stats['closed_trades'][0]['net_pnl'],0)
        self.assertEqual(result.win_rate,0)
        self.assertAlmostEqual(result.stats['ending_equity']-1000,sum(t['net_pnl'] for t in result.stats['closed_trades']))

    @unittest.skipUnless(PYTORCH_AVAILABLE,'PyTorch unavailable')
    def test_loss_has_finite_label_sensitive_gradients(self):
        import torch
        from quant_trade_system.core.statistical_learning_layer import CombinedLoss, OddsRatioLoss
        p=torch.tensor([.8,.8],requires_grad=True)
        CombinedLoss()(p,torch.tensor([1.,0.])).backward()
        self.assertLess(p.grad[0],0)
        self.assertGreater(p.grad[1],0)
        with self.assertRaises(ValueError): OddsRatioLoss()(p,torch.tensor([1.,0.]))

if __name__=='__main__': unittest.main()
