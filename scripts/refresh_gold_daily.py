"""Replace legacy demo gold bars with audited 2024-2026 REAL GC=F daily bars.

Source: Yahoo Finance public chart API, symbol GC=F (continuous nearby COMEX).
No synthetic fills; abort rather than silently patch missing dates or bad values.
Run on connected GitHub Actions; source url, sample audit and hashes preserved.
"""
from __future__ import annotations
import csv,datetime as dt,hashlib,json,random,urllib.request
from pathlib import Path

START=dt.date(2024,1,1)
END=dt.date(2026,10,30)
ROOT=Path(__file__).resolve().parents[1]
SOURCE="https://query1.finance.yahoo.com/v8/finance/chart/GC%3DF?period1=1704067200&period2=1793491200&interval=1d"

def fetch_json(url:str)->dict:
    errors=[]
    for host in ("query1.finance.yahoo.com","query2.finance.yahoo.com"):
        u=url.replace("query1.finance.yahoo.com",host)
        try:
            req=urllib.request.Request(u,headers={"User-Agent":"Mozilla/5.0 (compatible; QuantResearchDataAudit/1.0)","Accept":"application/json"})
            with urllib.request.urlopen(req,timeout=35) as r:
                return json.loads(r.read())
        except Exception as exc:
            errors.append(type(exc).__name__+":"+str(exc))
    raise RuntimeError("Cannot fetch authoritative daily series: "+"; ".join(errors))

def load()->list[dict]:
    result=fetch_json(SOURCE)["chart"]["result"][0]
    if result["meta"]["symbol"]!="GC=F" or result["meta"]["currency"]!="USD":
        raise RuntimeError("Unexpected instrument or quote currency")
    q=result["indicators"]["quote"][0]
    data=[]
    for i,t in enumerate(result["timestamp"]):
        day=dt.datetime.fromtimestamp(t,dt.timezone.utc).date()
        if not START<=day<=END: continue
        vals={key:q[key][i] for key in ("open","high","low","close","volume")}
        if any(x is None for x in vals.values()):continue
        rec={"timestamp":day.isoformat()}
        for k in ("open","high","low","close"):
            rec[k]=round(float(vals[k]),4)
        rec["volume"]=int(vals["volume"])
        data.append(rec)
    return data

def audit(rows:list[dict])->dict:
    days=[x["timestamp"] for x in rows]
    if len(rows)<690 or len(set(days))!=len(days) or days!=sorted(days):
        raise RuntimeError(f"coverage/duplicate/order failure rows={len(rows)}, unique={len(set(days))}")
    yr={str(y):sum(x["timestamp"].startswith(str(y)) for x in rows) for y in (2024,2025,2026)}
    if yr["2024"]<240 or yr["2025"]<240 or yr["2026"]<195:
        raise RuntimeError("Year coverage check failed: "+str(yr))
    if days[0]>"2024-01-03" or days[-1]!="2026-10-30":
        raise RuntimeError(f"Bad endpoints: {days[0]} -> {days[-1]}")
    gross_ohlc=[]
    for x in rows:
        if min(x[k] for k in ("open","high","low","close"))<=0 or x["high"]<x["low"] or x["volume"]<0:
            raise RuntimeError("Invalid OHLC/volume "+str(x))
        if x["high"]<max(x["open"],x["close"])*.95 or x["low"]>min(x["open"],x["close"])*1.05:
            gross_ohlc.append(x["timestamp"])
    if gross_ohlc:raise RuntimeError("Gross OHLC anomalies: "+str(gross_ohlc[:6]))
    prices={x["timestamp"]:x["close"] for x in rows}
    # Independently published reference values; nearby/settlement conventions vary.
    # 2026 anchors from Investing.com GC1! daily 2026-10-26..30.
    anchor=[
      ("2024-10-04",2667.8,0.025,"COMEX Dec24 indicative settlement / secondary report"),
      ("2025-12-31",4341.1,0.03,"Secondary source referencing GC=F Dec25 daily settlement"),
      ("2026-10-26",4038.4,0.015,"Investing.com GC1! published historical close"),
      ("2026-10-27",4074.6,0.015,"Investing.com GC1! published historical close"),
      ("2026-10-28",4051.6,0.015,"Investing.com GC1! published historical close"),
      ("2026-10-29",4031.4,0.015,"Reuters reported U.S. December futures settlement"),
      ("2026-10-30",3884.9,0.015,"Investing.com GC1! published historical close"),
    ]
    comparison=[]
    for d,reference,tolerance,src in anchor:
        if d not in prices:raise RuntimeError("Missing external sample "+d)
        err=abs(prices[d]-reference)/reference
        comparison.append({"date":d,"actual_gc":prices[d],"reference_gc":reference,
                           "relative_error_pct":round(err*100,4),"tolerance_pct":100*tolerance,"source":src})
        if err>tolerance:
            raise RuntimeError(f"Independent sample FAIL {d}: actual={prices[d]} expected~{reference}")
    rng=random.Random(20261031)
    internal={}
    for year in ("2024","2025","2026"):
        yr_rows=[x for x in rows if x["timestamp"].startswith(year)]
        internal[year]=[{"date":x["timestamp"],"close":x["close"],"volume":x["volume"]}
                        for x in rng.sample(yr_rows,min(5,len(yr_rows)))]
    # These internal random checks verify rows and basic OHLC only, not independent market truth.
    return {"status":"PASSED","instrument":"COMEX GC nearby continuous, Yahoo Finance GC=F",
            "currency":"USD","unit":"USD per troy oz","source_url":SOURCE,"start":days[0],"end":days[-1],
            "trading_rows":len(rows),"rows_by_year":yr,
            "independent_sample_checks":comparison,
            "internal_random_row_samples_NOT_independent_validation":internal,
            "warning":"Continuous nearby COMEX roll dates differ from fixed GCZ6; cannot substitute for AU2612."}

def main():
    rows=load()
    report=audit(rows)
    p=ROOT/"data"/"gold_daily.csv"
    p.parent.mkdir(parents=True,exist_ok=True)
    # Preserve legacy synthetic/unverified file for reproducibility, do NOT mix with real.
    archive=ROOT/"data"/"archive"/"gold_daily_legacy_UNVERIFIED.csv"
    if p.exists() and not archive.exists():
        archive.parent.mkdir(parents=True,exist_ok=True)
        archive.write_bytes(p.read_bytes())
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["timestamp","open","high","low","close","volume"])
        w.writeheader();w.writerows(rows)
    report["file_sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/"data"/"gold_daily_provenance.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("AUDITED_GC_HISTORY",json.dumps({k:v for k,v in report.items() if k!="internal_random_row_samples_NOT_independent_validation"},ensure_ascii=False))

if __name__=="__main__":main()
