from __future__ import annotations

import json
import math
import sys
from typing import Any, Dict


def _cmd_status() -> Dict[str, Any]:
    import QuantLib as ql  # noqa: WPS433

    return {"version": getattr(ql, "__version__", "unknown"), "python": sys.version.split()[0]}


def _cmd_black_scholes(payload: Dict[str, Any]) -> Dict[str, Any]:
    import QuantLib as ql  # noqa: WPS433

    spot = float(payload.get("spot", 100))
    strike = float(payload.get("strike", 100))
    volatility = float(payload.get("volatility", 0.2))
    risk_free_rate = float(payload.get("risk_free_rate", 0.03))
    maturity_days = int(payload.get("maturity_days", 30))
    option_type = str(payload.get("option_type", "call")).lower()

    calculation_date = ql.Date.todaysDate()
    maturity_date = calculation_date + maturity_days
    ql.Settings.instance().evaluationDate = calculation_date

    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put,
        strike,
    )
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, ql.Actual365Fixed()))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, 0.0, ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(calculation_date, ql.NullCalendar(), volatility, ql.Actual365Fixed())
    )
    process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, flat_ts, vol_ts)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    return {
        "npv": round(float(option.NPV()), 6),
        "delta": round(float(option.delta()), 6),
        "gamma": round(float(option.gamma()), 6),
        "vega": round(float(option.vega()), 6),
        "theta": round(float(option.theta()), 6) if not math.isnan(option.theta()) else None,
        "rho": round(float(option.rho()), 6),
    }


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    payload = {}
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        if raw:
            payload = json.loads(raw)
    commands = {
        "status": _cmd_status,
        "black_scholes": lambda: _cmd_black_scholes(payload),
    }
    try:
        data = commands[command]()
        print(json.dumps({"ok": True, "data": data}, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
