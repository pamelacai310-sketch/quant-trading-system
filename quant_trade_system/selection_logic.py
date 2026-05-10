from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


class DailySelectionEngine:
    """Daily selection rules for equities and futures."""

    NASDAQ_MARKET_CODE = "105"
    NON_OTC_MARKET_CODES = {"105", "106", "107"}
    MARKET_LABELS = {
        "105": "NASDAQ",
        "106": "NYSE",
        "107": "AMEX",
        "153": "OTC",
    }

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).replace(",", "").replace("%", "").strip()
        if text in {"", "-", "--", "None", "nan"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @classmethod
    def _compact_anchor_date(cls, value: Optional[str]) -> str:
        text = (value or "").strip()
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) == 8:
            return digits
        return datetime.now().strftime("%Y%m%d")

    @classmethod
    def _normalize_us_spot_frame(cls, frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "market_code",
                    "exchange",
                    "code",
                    "name",
                    "latest_price",
                    "change_pct",
                    "turnover",
                    "net_inflow",
                ]
            )

        working = frame.copy()
        if "代码" not in working.columns and {"f12", "f13"}.issubset(set(working.columns)):
            working["代码"] = working["f13"].astype(str) + "." + working["f12"].astype(str)
        if "名称" not in working.columns and "f14" in working.columns:
            working["名称"] = working["f14"]
        if "简称" not in working.columns and "f12" in working.columns:
            working["简称"] = working["f12"]
        if "编码" not in working.columns and "f13" in working.columns:
            working["编码"] = working["f13"]
        if "最新价" not in working.columns and "f2" in working.columns:
            working["最新价"] = working["f2"]
        if "涨跌幅" not in working.columns and "f3" in working.columns:
            working["涨跌幅"] = working["f3"]
        if "成交额" not in working.columns and "f6" in working.columns:
            working["成交额"] = working["f6"]
        if "主力净流入-净额" not in working.columns and "f62" in working.columns:
            working["主力净流入-净额"] = working["f62"]

        working["code"] = working["代码"].astype(str)
        working["ticker"] = working["code"].str.split(".").str[-1]
        working["market_code"] = working["code"].str.split(".").str[0]
        working["exchange"] = working["market_code"].map(cls.MARKET_LABELS).fillna("UNKNOWN")
        working["name"] = working.get("名称", working.get("name", working["ticker"])).astype(str)
        working["latest_price"] = pd.to_numeric(
            working.get("最新价", working.get("latest_price")),
            errors="coerce",
        )
        working["change_pct"] = pd.to_numeric(
            working.get("涨跌幅", working.get("change_pct")),
            errors="coerce",
        )
        working["turnover"] = pd.to_numeric(
            working.get("成交额", working.get("turnover")),
            errors="coerce",
        )
        working["net_inflow"] = pd.to_numeric(
            working.get("主力净流入-净额", working.get("net_inflow")),
            errors="coerce",
        )
        return working

    @staticmethod
    def _normalize_us_hist_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["date", "close"])
        working = frame.copy()
        rename_map = {}
        for column in working.columns:
            lower = column.lower()
            if lower in {"date", "日期"}:
                rename_map[column] = "date"
            elif lower in {"close", "收盘"}:
                rename_map[column] = "close"
        working = working.rename(columns=rename_map)
        if not {"date", "close"}.issubset(set(working.columns)):
            return pd.DataFrame(columns=["date", "close"])
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        working["close"] = pd.to_numeric(working["close"], errors="coerce")
        working = working.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return working[["date", "close"]]

    @classmethod
    def rank_nasdaq_net_inflow(
        cls,
        frame: pd.DataFrame,
        topn: int = 20,
    ) -> Dict[str, Any]:
        working = cls._normalize_us_spot_frame(frame)
        working = working[working["market_code"] == cls.NASDAQ_MARKET_CODE].copy()
        working = working.dropna(subset=["net_inflow", "latest_price"])
        working = working.sort_values("net_inflow", ascending=False).head(topn)
        records = [
            {
                "ticker": row["ticker"],
                "code": row["code"],
                "name": row["name"],
                "exchange": row["exchange"],
                "latest_price": round(float(row["latest_price"]), 4),
                "change_pct": round(float(row["change_pct"]), 4) if pd.notna(row["change_pct"]) else None,
                "net_inflow": round(float(row["net_inflow"]), 4),
                "turnover": round(float(row["turnover"]), 4) if pd.notna(row["turnover"]) else None,
            }
            for _, row in working.iterrows()
        ]
        return {
            "criteria": {
                "exchange": "NASDAQ",
                "exclude_otc": True,
                "sort_by": "net_inflow_desc",
            },
            "records": records,
            "count": len(records),
        }

    @classmethod
    def rank_china_futures_open_interest(
        cls,
        frame: pd.DataFrame,
        topn: int = 20,
    ) -> Dict[str, Any]:
        if frame is None or frame.empty:
            return {
                "criteria": {
                    "sort_by": "top20_member_open_interest_proxy_desc",
                    "proxy_formula": "long_open_interest_top20 + short_open_interest_top20",
                },
                "records": [],
                "count": 0,
            }
        working = frame.copy()
        for column in [
            "vol_top20",
            "long_open_interest_top20",
            "short_open_interest_top20",
        ]:
            if column in working.columns:
                working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)
            else:
                working[column] = 0.0
        working["open_interest_proxy"] = (
            working["long_open_interest_top20"] + working["short_open_interest_top20"]
        )
        working = working.sort_values(
            ["open_interest_proxy", "vol_top20"],
            ascending=False,
        ).head(topn)
        records = [
            {
                "contract": str(row.get("symbol", "")),
                "variety": str(row.get("var", "")),
                "open_interest_proxy": round(float(row["open_interest_proxy"]), 4),
                "long_open_interest_top20": round(float(row["long_open_interest_top20"]), 4),
                "short_open_interest_top20": round(float(row["short_open_interest_top20"]), 4),
                "vol_top20": round(float(row["vol_top20"]), 4),
            }
            for _, row in working.iterrows()
        ]
        return {
            "criteria": {
                "sort_by": "top20_member_open_interest_proxy_desc",
                "proxy_formula": "long_open_interest_top20 + short_open_interest_top20",
            },
            "records": records,
            "count": len(records),
        }

    @classmethod
    def screen_us_ytd_hot_stocks(
        cls,
        spot_frame: pd.DataFrame,
        history_loader: Callable[[str, str], pd.DataFrame],
        pink_frame: Optional[pd.DataFrame] = None,
        anchor_date: Optional[str] = None,
        min_price: float = 100.0,
        min_ytd_return: float = 0.50,
        topn: int = 50,
    ) -> Dict[str, Any]:
        anchor_compact = cls._compact_anchor_date(anchor_date)
        anchor_ts = pd.Timestamp(datetime.strptime(anchor_compact, "%Y%m%d"))
        year_start = f"{anchor_ts.year}0101"

        spot = cls._normalize_us_spot_frame(spot_frame)
        spot = spot[spot["market_code"].isin(cls.NON_OTC_MARKET_CODES)].copy()
        spot = spot[spot["latest_price"] > min_price].copy()

        pink = cls._normalize_us_spot_frame(pink_frame) if pink_frame is not None else pd.DataFrame()
        pink_tickers = set(pink["ticker"].dropna().astype(str)) if not pink.empty else set()
        pink_codes = set(pink["code"].dropna().astype(str)) if not pink.empty else set()
        if pink_tickers or pink_codes:
            spot = spot[
                ~spot["ticker"].astype(str).isin(pink_tickers)
                & ~spot["code"].astype(str).isin(pink_codes)
            ].copy()

        records: List[Dict[str, Any]] = []
        for _, row in spot.iterrows():
            history = cls._normalize_us_hist_frame(history_loader(str(row["code"]), year_start))
            if history.empty:
                continue
            history = history[history["date"] <= anchor_ts]
            if history.empty:
                continue
            first_close = float(history["close"].iloc[0])
            last_close = float(history["close"].iloc[-1])
            if first_close <= 0:
                continue
            ytd_return = last_close / first_close - 1.0
            if ytd_return <= min_ytd_return:
                continue
            records.append(
                {
                    "ticker": row["ticker"],
                    "code": row["code"],
                    "name": row["name"],
                    "exchange": row["exchange"],
                    "latest_price": round(float(row["latest_price"]), 4),
                    "ytd_return_pct": round(ytd_return * 100.0, 4),
                    "ytd_return": round(ytd_return, 6),
                    "first_close_ytd": round(first_close, 4),
                    "last_close_ytd": round(last_close, 4),
                }
            )
        records.sort(key=lambda item: item["ytd_return"], reverse=True)
        return {
            "criteria": {
                "markets": ["NASDAQ", "NYSE", "AMEX"],
                "exclude_otc": True,
                "anchor_close_date": anchor_compact,
                "min_price": min_price,
                "min_ytd_return": min_ytd_return,
                "price_basis": "latest_close",
                "return_basis": "qfq_adjusted_close",
            },
            "records": records[:topn],
            "count": min(len(records), topn),
            "otc_excluded": True,
        }

    @staticmethod
    def extract_default_watchlist_symbols(
        selection_logic: Dict[str, Any],
        limit: int = 8,
    ) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for section_name in ["nasdaq_top_net_inflow", "us_ytd_hot_non_otc"]:
            for item in selection_logic.get(section_name, {}).get("records", []):
                ticker = str(item.get("ticker", "")).strip().upper()
                if not ticker or ticker in seen:
                    continue
                ordered.append(ticker)
                seen.add(ticker)
                if len(ordered) >= limit:
                    return ordered
        return ordered
