"""Explicit 2025–2026 exchange sessions, never inferred from quote availability.

Sources are exchange annual closure notices. Exceptional future closures still
require an updated calendar. Weekend make-up workdays are not exchange sessions.
"""
import pandas as pd

SOURCES = {
    "2025": "https://www.shfe.com.cn/eng/CircularNews/Circular/202412/t20241223_827504.html",
    "2026": "https://www.cffex.com.cn/en_new/Notices/20251217/46426.html",
}
CLOSURES = [
    ("2025-01-01", "2025-01-01"), ("2025-01-28", "2025-02-04"),
    ("2025-04-04", "2025-04-06"), ("2025-05-01", "2025-05-05"),
    ("2025-05-31", "2025-06-02"), ("2025-10-01", "2025-10-08"),
    ("2026-01-01", "2026-01-03"), ("2026-02-15", "2026-02-23"),
    ("2026-04-04", "2026-04-06"), ("2026-05-01", "2026-05-05"),
    ("2026-06-19", "2026-06-21"), ("2026-09-25", "2026-09-27"),
    ("2026-10-01", "2026-10-07"),
]


def sessions(start: str, end: str):
    if start < "2025-01-01" or end > "2026-12-31" or start > end:
        raise ValueError("Calendar coverage is 2025-01-01 through 2026-12-31")
    return [str(day) for day in pd.bdate_range(start, end).strftime("%Y-%m-%d")
            if not any(first <= day <= last for first, last in CLOSURES)]
