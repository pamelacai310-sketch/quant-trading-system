"""Conservative research-output overlay; never creates broker orders.

Empirical outcome frequencies are diagnostics, not calibrated probabilities.
The fixed deadband is a policy hypothesis requiring prospective validation.
"""
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean


@dataclass(frozen=True)
class DirectionEvidencePolicy:
    min_dates: int = 30
    history_dates: int = 60
    volatility_deadband: float = 0.10
    edge_buffer: float = 0.0002

    def __post_init__(self):
        if not 1 <= self.min_dates <= self.history_dates:
            raise ValueError("Invalid history windows")
        if any(not isfinite(v) or v < 0 for v in
               (self.volatility_deadband, self.edge_buffer)):
            raise ValueError("Invalid thresholds")


def assess_direction(row, records, *, cost=None, forward_validated=False,
                     gate_passed=False, policy=DirectionEvidencePolicy()):
    """Filter completed, same-contract, same-sign one-session predictions.

    Input dates must be canonical ISO dates from the structural replay. Cost is
    a round-trip decimal return estimate, not a score or margin percentage.
    """
    result = {'symbol': row['symbol'], 'asof': row['feature_date'],
              'direction': 'NEUTRAL', 'action': 'OBSERVE_ONLY',
              'version': 'direction_evidence_v1', 'reasons': [],
              'probability_calibrated': False, 'cost': cost}
    pred, sigma = row.get('prediction'), row.get('sigma')
    if any(v is None or not isfinite(v) for v in (pred, sigma)) or sigma <= 0:
        result['reasons'] = ['invalid_prediction']; return result
    sign = 1 if pred > 0 else -1 if pred < 0 else 0
    result['raw_direction'] = 'LONG' if sign > 0 else 'SHORT' if sign < 0 else 'NEUTRAL'
    valid_cost = cost is not None and isfinite(cost) and cost >= 0
    band = max(policy.volatility_deadband*sigma,
               cost+policy.edge_buffer if valid_cost else 0)
    result.update(expected_return=pred, deadband=band)
    if abs(pred) > band:
        result['direction'] = result['raw_direction']
    else:
        result['reasons'].append('edge_below_noise_or_cost')
    history = {}
    for r in records:
        target = r.get('target_date')
        if (r['symbol'] != row['symbol'] or not target or
                not r['feature_date'] < target <= row['feature_date']):
            continue
        actual, forecast = r.get('actual'), r.get('prediction')
        if any(v is None or not isfinite(v) for v in (actual, forecast)):
            continue
        if forecast*sign <= 0:
            continue
        if target in history:
            raise ValueError('Duplicate contract/target outcomes')
        history[target] = sign*actual
    returns = [history[d] for d in sorted(history)[-policy.history_dates:]]
    result['sample_dates'] = len(returns)
    if not valid_cost:
        result['reasons'].append('execution_cost_unavailable')
    if len(returns) < policy.min_dates:
        result['reasons'].append('insufficient_completed_outcomes')
    elif valid_cost:
        net = [r-cost for r in returns]
        wins, losses = [r for r in net if r>0], [-r for r in net if r<0]
        n=len(net); p=len(wins)/n; z=1.96
        lower=(p+z*z/(2*n)-z*sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
        result.update(empirical_net_win_rate=p, win_rate_lower_diagnostic=lower,
                      empirical_payoff_ratio=mean(wins)/mean(losses) if wins and losses else None,
                      empirical_net_mean=mean(net),
                      uncertainty_note='Wilson is diagnostic; serial dependence is not corrected')
        if not wins or not losses:
            result['reasons'].append('payoff_distribution_incomplete')
        if mean(net)<=0 or lower<=0.5:
            result['reasons'].append('historical_edge_not_established')
    if not forward_validated or not gate_passed:
        result['reasons'].append('model_not_validated')
    if not result['reasons'] and result['direction'] != 'NEUTRAL':
        result['action'] = result['direction']
    return result
