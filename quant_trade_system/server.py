from __future__ import annotations

import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
from urllib.parse import parse_qs, urlparse

from .service import QuantTradingService


class QuantRequestHandler(SimpleHTTPRequestHandler):
    service: QuantTradingService
    static_dir: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=self.static_dir, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api("GET", parsed.path, parse_qs(parsed.query), None)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON payload"})
            return
        parsed = urlparse(self.path)
        self._handle_api("POST", parsed.path, parse_qs(parsed.query), payload)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle_api(self, method: str, path: str, query: Dict[str, Any], payload: Dict[str, Any] | None) -> None:
        routes: Dict[Tuple[str, str], Callable[[Dict[str, Any], Dict[str, Any]], Any]] = {
            ("GET", "/api/health"): lambda _query, _payload: {"status": "ok"},
            ("GET", "/api/dashboard"): lambda _query, _payload: self.service.dashboard(),
            ("GET", "/api/strategies"): lambda _query, _payload: self.service.list_strategies(),
            ("POST", "/api/strategies"): lambda _query, body: self.service.save_strategy(body),
            ("POST", "/api/backtest"): lambda _query, body: self.service.backtest_strategy(body["strategy_id"]),
            ("GET", "/api/backtests"): lambda q, _payload: self.service.list_backtests(q.get("strategy_id", [None])[0]),
            ("POST", "/api/orders"): lambda _query, body: self.service.submit_order(body),
            ("GET", "/api/orders"): lambda _query, _payload: self.service.storage.list_orders(),
            ("POST", "/api/execute"): lambda _query, body: self.service.execute_strategy(
                body["strategy_id"], body.get("broker_mode", "paper")
            ),
            ("GET", "/api/research"): lambda _query, _payload: self.service.research_summary(),
            ("GET", "/api/datasets"): lambda _query, _payload: self.service.list_datasets(),
            ("GET", "/api/data/series"): lambda q, _payload: self.service.dataset_series(q.get("dataset", ["gold_daily"])[0]),
            ("GET", "/api/causal/status"): lambda _query, _payload: self.service.causal_status(),
            ("POST", "/api/causal/pipeline"): lambda _query, body: self.service.run_causal_pipeline(body.get("symbols")),
            ("GET", "/api/causal/market"): lambda _query, _payload: self.service.get_market_regime_snapshot(),
            ("GET", "/api/causal/decision"): lambda _query, _payload: self.service.get_causal_decision(),
            ("POST", "/api/causal/decision"): lambda _query, body: self.service.get_causal_decision(body.get("market_data")),
            ("POST", "/api/causal/execute"): lambda _query, body: self.service.execute_causal_decision(
                body.get("broker_mode", "paper"), body.get("market_data")
            ),
            ("GET", "/api/ecosystem/status"): lambda _query, _payload: self.service.ecosystem_status(),
            ("POST", "/api/backtest/advanced"): lambda _query, body: self.service.advanced_backtest(body["strategy_id"]),
            ("POST", "/api/export/strategy"): lambda _query, body: self.service.export_strategy(body["strategy_id"], body["target"]),
            ("POST", "/api/options/price"): lambda _query, body: self.service.price_option(body),
        }
        handler = routes.get((method, path))
        if handler is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Route {path} not found"})
            return
        try:
            result = handler(query, payload or {})
        except Exception as exc:  # noqa: BLE001
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, result)

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(base_dir: str, host: str = "127.0.0.1", port: int = 8108) -> None:
    service = QuantTradingService(base_dir)
    static_dir = str(Path(base_dir) / "static")

    class Handler(QuantRequestHandler):
        pass

    Handler.service = service
    Handler.static_dir = static_dir
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Causal AI quant trading system running at http://{host}:{port}")
    server.serve_forever()
