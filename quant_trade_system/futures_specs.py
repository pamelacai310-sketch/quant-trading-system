"""Chinese futures contract specifications used for margin math.

The key production rule is:

    margin = latest_price * contract_multiplier * lots * margin_rate

Prices for Chinese commodity futures are usually quoted per ton, gram, barrel
or index point.  The contract multiplier converts one quoted price point into
one lot's notional value.  Brokerage platforms may apply a higher margin rate
than the exchange minimum, so callers should pass the platform rate when it is
known.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class FuturesContractSpec:
    symbol: str
    exchange: str
    name: str
    multiplier: float
    unit: str
    exchange_min_margin_rate: float


CN_FUTURES_CONTRACT_SPECS: Dict[str, FuturesContractSpec] = {
    # SHFE / INE products currently used by the project universe.
    "CU": FuturesContractSpec("CU", "SHFE", "阴极铜", 5.0, "吨/手", 0.05),
    "BC": FuturesContractSpec("BC", "INE", "国际铜", 5.0, "吨/手", 0.05),
    "AL": FuturesContractSpec("AL", "SHFE", "铝", 5.0, "吨/手", 0.05),
    "ZN": FuturesContractSpec("ZN", "SHFE", "锌", 5.0, "吨/手", 0.05),
    "PB": FuturesContractSpec("PB", "SHFE", "铅", 5.0, "吨/手", 0.05),
    "NI": FuturesContractSpec("NI", "SHFE", "镍", 1.0, "吨/手", 0.05),
    "SN": FuturesContractSpec("SN", "SHFE", "锡", 1.0, "吨/手", 0.05),
    "AO": FuturesContractSpec("AO", "SHFE", "氧化铝", 20.0, "吨/手", 0.05),
    "AD": FuturesContractSpec("AD", "SHFE", "铸造铝合金", 5.0, "吨/手", 0.05),
    "AU": FuturesContractSpec("AU", "SHFE", "黄金", 1000.0, "克/手", 0.04),
    "AG": FuturesContractSpec("AG", "SHFE", "白银", 15.0, "千克/手", 0.04),
    "RB": FuturesContractSpec("RB", "SHFE", "螺纹钢", 10.0, "吨/手", 0.05),
    "WR": FuturesContractSpec("WR", "SHFE", "线材", 10.0, "吨/手", 0.05),
    "HC": FuturesContractSpec("HC", "SHFE", "热轧卷板", 10.0, "吨/手", 0.04),
    "SS": FuturesContractSpec("SS", "SHFE", "不锈钢", 5.0, "吨/手", 0.05),
    "SC": FuturesContractSpec("SC", "INE", "原油", 1000.0, "桶/手", 0.05),
    "LU": FuturesContractSpec("LU", "INE", "低硫燃料油", 10.0, "吨/手", 0.05),
    "FU": FuturesContractSpec("FU", "SHFE", "燃料油", 10.0, "吨/手", 0.08),
    "BU": FuturesContractSpec("BU", "SHFE", "石油沥青", 10.0, "吨/手", 0.08),
    "BR": FuturesContractSpec("BR", "SHFE", "合成橡胶", 5.0, "吨/手", 0.08),
    "RU": FuturesContractSpec("RU", "SHFE", "天然橡胶", 10.0, "吨/手", 0.05),
    "NR": FuturesContractSpec("NR", "INE", "20号胶", 10.0, "吨/手", 0.05),
    "SP": FuturesContractSpec("SP", "SHFE", "纸浆", 10.0, "吨/手", 0.05),
    "OP": FuturesContractSpec("OP", "SHFE", "胶版印刷纸", 20.0, "吨/手", 0.05),
    "EC": FuturesContractSpec("EC", "INE", "集运指数欧线", 50.0, "元/点", 0.12),
    # DCE products.
    "A": FuturesContractSpec("A", "DCE", "豆一", 10.0, "吨/手", 0.05),
    "B": FuturesContractSpec("B", "DCE", "豆二", 10.0, "吨/手", 0.05),
    "M": FuturesContractSpec("M", "DCE", "豆粕", 10.0, "吨/手", 0.05),
    "Y": FuturesContractSpec("Y", "DCE", "豆油", 10.0, "吨/手", 0.05),
    "P": FuturesContractSpec("P", "DCE", "棕榈油", 10.0, "吨/手", 0.05),
    "C": FuturesContractSpec("C", "DCE", "玉米", 10.0, "吨/手", 0.05),
    "CS": FuturesContractSpec("CS", "DCE", "玉米淀粉", 10.0, "吨/手", 0.05),
    "JD": FuturesContractSpec("JD", "DCE", "鸡蛋", 10.0, "500千克/手报价乘数", 0.05),
    "LH": FuturesContractSpec("LH", "DCE", "生猪", 16.0, "吨/手", 0.08),
    "L": FuturesContractSpec("L", "DCE", "聚乙烯", 5.0, "吨/手", 0.05),
    "V": FuturesContractSpec("V", "DCE", "PVC", 5.0, "吨/手", 0.05),
    "PP": FuturesContractSpec("PP", "DCE", "聚丙烯", 5.0, "吨/手", 0.05),
    "EG": FuturesContractSpec("EG", "DCE", "乙二醇", 10.0, "吨/手", 0.06),
    "EB": FuturesContractSpec("EB", "DCE", "苯乙烯", 5.0, "吨/手", 0.07),
    "PG": FuturesContractSpec("PG", "DCE", "液化石油气", 20.0, "吨/手", 0.07),
    "I": FuturesContractSpec("I", "DCE", "铁矿石", 100.0, "吨/手", 0.08),
    "J": FuturesContractSpec("J", "DCE", "焦炭", 100.0, "吨/手", 0.08),
    "JM": FuturesContractSpec("JM", "DCE", "焦煤", 60.0, "吨/手", 0.08),
    "FB": FuturesContractSpec("FB", "DCE", "纤维板", 10.0, "立方米/手", 0.10),
    "BB": FuturesContractSpec("BB", "DCE", "胶合板", 500.0, "张/手", 0.10),
    "RR": FuturesContractSpec("RR", "DCE", "粳米", 10.0, "吨/手", 0.05),
    # CZCE products.
    "CF": FuturesContractSpec("CF", "CZCE", "棉花", 5.0, "吨/手", 0.05),
    "SR": FuturesContractSpec("SR", "CZCE", "白糖", 10.0, "吨/手", 0.05),
    "TA": FuturesContractSpec("TA", "CZCE", "PTA", 5.0, "吨/手", 0.06),
    "OI": FuturesContractSpec("OI", "CZCE", "菜油", 10.0, "吨/手", 0.05),
    "RM": FuturesContractSpec("RM", "CZCE", "菜粕", 10.0, "吨/手", 0.05),
    "MA": FuturesContractSpec("MA", "CZCE", "甲醇", 10.0, "吨/手", 0.07),
    "FG": FuturesContractSpec("FG", "CZCE", "玻璃", 20.0, "吨/手", 0.07),
    "SA": FuturesContractSpec("SA", "CZCE", "纯碱", 20.0, "吨/手", 0.07),
    "UR": FuturesContractSpec("UR", "CZCE", "尿素", 20.0, "吨/手", 0.05),
    "PF": FuturesContractSpec("PF", "CZCE", "短纤", 5.0, "吨/手", 0.07),
    "PK": FuturesContractSpec("PK", "CZCE", "花生", 5.0, "吨/手", 0.08),
    "AP": FuturesContractSpec("AP", "CZCE", "苹果", 10.0, "吨/手", 0.08),
    "CJ": FuturesContractSpec("CJ", "CZCE", "红枣", 5.0, "吨/手", 0.07),
    "CY": FuturesContractSpec("CY", "CZCE", "棉纱", 5.0, "吨/手", 0.05),
    "PM": FuturesContractSpec("PM", "CZCE", "普麦", 50.0, "吨/手", 0.05),
    "WH": FuturesContractSpec("WH", "CZCE", "强麦", 20.0, "吨/手", 0.05),
    "JR": FuturesContractSpec("JR", "CZCE", "粳稻", 20.0, "吨/手", 0.05),
    "RI": FuturesContractSpec("RI", "CZCE", "早籼稻", 20.0, "吨/手", 0.05),
    "LR": FuturesContractSpec("LR", "CZCE", "晚籼稻", 20.0, "吨/手", 0.05),
    "RS": FuturesContractSpec("RS", "CZCE", "菜籽", 10.0, "吨/手", 0.05),
    "SF": FuturesContractSpec("SF", "CZCE", "硅铁", 5.0, "吨/手", 0.07),
    "SM": FuturesContractSpec("SM", "CZCE", "锰硅", 5.0, "吨/手", 0.07),
    "SH": FuturesContractSpec("SH", "CZCE", "烧碱", 30.0, "吨/手", 0.08),
    "PX": FuturesContractSpec("PX", "CZCE", "对二甲苯", 5.0, "吨/手", 0.08),
    "PR": FuturesContractSpec("PR", "CZCE", "瓶片", 15.0, "吨/手", 0.08),
    "ZC": FuturesContractSpec("ZC", "CZCE", "动力煤", 100.0, "吨/手", 0.08),
    # CFFEX products. Stock index futures are yuan/index-point multipliers;
    # treasury futures use RMB 1,000,000 face value, i.e. quote point * 10,000.
    "IF": FuturesContractSpec("IF", "CFFEX", "沪深300股指", 300.0, "元/点", 0.08),
    "IH": FuturesContractSpec("IH", "CFFEX", "上证50股指", 300.0, "元/点", 0.08),
    "IC": FuturesContractSpec("IC", "CFFEX", "中证500股指", 200.0, "元/点", 0.10),
    "IM": FuturesContractSpec("IM", "CFFEX", "中证1000股指", 200.0, "元/点", 0.10),
    "T": FuturesContractSpec("T", "CFFEX", "10年期国债", 10000.0, "元/点", 0.02),
    "TF": FuturesContractSpec("TF", "CFFEX", "5年期国债", 10000.0, "元/点", 0.012),
    "TS": FuturesContractSpec("TS", "CFFEX", "2年期国债", 20000.0, "元/点", 0.005),
    "TL": FuturesContractSpec("TL", "CFFEX", "30年期国债", 10000.0, "元/点", 0.035),
    # GFEX products.
    "SI": FuturesContractSpec("SI", "GFEX", "工业硅", 5.0, "吨/手", 0.08),
    "LC": FuturesContractSpec("LC", "GFEX", "碳酸锂", 1.0, "吨/手", 0.09),
    "PS": FuturesContractSpec("PS", "GFEX", "多晶硅", 3.0, "吨/手", 0.08),
}


def normalize_futures_symbol(symbol: str) -> str:
    """Return the product code from a contract code such as CU2607 or cu0."""
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ""
    match = re.match(r"([A-Z]+)", raw)
    return match.group(1) if match else raw


def get_futures_contract_spec(symbol: str) -> Optional[FuturesContractSpec]:
    return CN_FUTURES_CONTRACT_SPECS.get(normalize_futures_symbol(symbol))


def futures_contract_multiplier(symbol: str, default: float = 1.0) -> float:
    spec = get_futures_contract_spec(symbol)
    return spec.multiplier if spec else default


def futures_exchange_min_margin_rate(symbol: str, default: float = 0.15) -> float:
    spec = get_futures_contract_spec(symbol)
    return spec.exchange_min_margin_rate if spec else default


def calculate_futures_notional(
    symbol: str,
    latest_price: float,
    lots: int = 1,
    multiplier: Optional[float] = None,
) -> float:
    effective_multiplier = futures_contract_multiplier(symbol) if multiplier is None else float(multiplier)
    return float(latest_price) * effective_multiplier * int(lots)


def calculate_futures_margin(
    symbol: str,
    latest_price: float,
    lots: int = 1,
    margin_rate: Optional[float] = None,
    multiplier: Optional[float] = None,
) -> float:
    effective_margin_rate = futures_exchange_min_margin_rate(symbol) if margin_rate is None else float(margin_rate)
    return calculate_futures_notional(symbol, latest_price, lots=lots, multiplier=multiplier) * effective_margin_rate


def build_one_lot_margin_table(
    latest_price_by_symbol: Mapping[str, float],
    margin_rate_by_symbol: Optional[Mapping[str, float]] = None,
) -> list[dict[str, float | str]]:
    """Build an auditable one-lot margin table for futures contracts.

    ``latest_price_by_symbol`` may contain product symbols (CU) or concrete
    contract symbols (CU2607). ``margin_rate_by_symbol`` should contain the
    broker/platform margin rate when it differs from the exchange minimum.
    """
    margin_rate_by_symbol = margin_rate_by_symbol or {}
    rows: list[dict[str, float | str]] = []
    for symbol, latest_price in latest_price_by_symbol.items():
        product = normalize_futures_symbol(symbol)
        spec = get_futures_contract_spec(product)
        multiplier = spec.multiplier if spec else 1.0
        margin_rate = (
            margin_rate_by_symbol.get(symbol)
            or margin_rate_by_symbol.get(product)
            or futures_exchange_min_margin_rate(product)
        )
        one_lot_margin = calculate_futures_margin(
            product,
            latest_price,
            lots=1,
            margin_rate=margin_rate,
            multiplier=multiplier,
        )
        rows.append(
            {
                "symbol": str(symbol),
                "product": product,
                "exchange": spec.exchange if spec else "UNKNOWN",
                "name": spec.name if spec else product,
                "latest_price": float(latest_price),
                "contract_multiplier": float(multiplier),
                "margin_rate": float(margin_rate),
                "one_lot_notional": calculate_futures_notional(product, latest_price, lots=1, multiplier=multiplier),
                "one_lot_margin": one_lot_margin,
            }
        )
    return rows
