from __future__ import annotations

import argparse
from pathlib import Path

from quant_trade_system.server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Causal AI quant trading system server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8108)
    args = parser.parse_args()
    run_server(str(Path(__file__).resolve().parent), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
