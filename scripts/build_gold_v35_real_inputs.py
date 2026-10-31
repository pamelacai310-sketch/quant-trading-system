"""Build a point-in-time gold/V3.5 backtest from REAL external data.

No dummy data. GC=F source was independently audited before being committed.
Everything else is fetched fresh on GitHub Actions. No raw GLD archive is
republished; sponsor terms prohibit unauthorized redistributions.
"""
from __future__ import annotations
import io,zipfile,json,time,urllib.parse,urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
END=pd.Timestamp("2026-10-30")
BEGIN=pd.Timestamp("2024-01-01")
REQ_HEADERS={"User-Agent":"Mozilla/5.0 (compatible; GoldResearchAudit/1.0)","Accept":"*/*"}
EM_PARAMS={"klt":"101","fqt":"1","lmt":"10000","end":"20500000","iscca":"1",
           "fields1":"f1,f2,f3,f4,f5,f6,f7,f8",
           "fields2":"f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64",
           "ut":"7eea3edcaed734bea9cbfc24409ed989","forcect":"1"}

def get(url:str,tries:int=3)->bytes:
    for k in range(tries):
        try:
            req=urllib.request.Request(url,headers=REQ_HEADERS)
            with urllib.request.urlopen(req,timeout=40) as r:
                b=r.read()
                if not b:raise RuntimeError('EMPTY')
                return b
        except Exception as e:
            print('RETRY',url.split('?')[0],type(e).__name__,str(e)[:100])
            if k==tries-1:raise
            time.sleep(k+1)
    raise AssertionError('unreachable')

def get_json(url:str):
    return json.loads(get(url).decode('utf-8-sig'))

def yahoo_dxy()->pd.DataFrame:
    url="https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?period1=1704067200&period2=1793491200&interval=1d"
    d=get_json(url)['chart']['result'][0]
    q=d['indicators']['quote'][0]
    frame=pd.DataFrame({'date':pd.to_datetime(d['timestamp'],unit='s',utc=True).date,
                        'dxy':q['close']})
    frame['date']=pd.to_datetime(frame['date'])
    return frame.dropna().drop_duplicates('date',keep='last').sort_values('date')

def fred_tips()->pd.DataFrame:
    url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10&cosd=2023-12-01&coed=2026-10-30"
    f=pd.read_csv(io.BytesIO(get(url)))
    datecol=f.columns[0]
    f=f.rename(columns={datecol:'date','DFII10':'tips10_pp'})
    f['date']=pd.to_datetime(f['date'])
    f['tips10_pp']=pd.to_numeric(f['tips10_pp'],errors='coerce')
    return f.dropna().sort_values('date')

def cot_releases()->pd.DataFrame:
    parts=[]
    for year in (2023,2024,2025,2026):
        url=f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
        z=zipfile.ZipFile(io.BytesIO(get(url)))
        f=pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),low_memory=False)
        name=f['Market_and_Exchange_Names'].astype(str).str.strip()
        gold=f.loc[name.str.startswith('GOLD - COMMODITY EXCHANGE INC.')].copy()
        if gold.empty:raise ValueError(f'CFTC gold contract missing in {year}')
        parts.append(gold)
    df=pd.concat(parts,ignore_index=True)
    df['report_date']=pd.to_datetime(df['Report_Date_as_YYYY-MM-DD'],errors='coerce')
    df['long']=pd.to_numeric(df['M_Money_Positions_Long_All'],errors='coerce')
    df['short']=pd.to_numeric(df['M_Money_Positions_Short_All'],errors='coerce')
    df['cot_mm_net']=df['long']-df['short']
    df['cot_open_interest']=pd.to_numeric(df['Open_Interest_All'],errors='coerce')
    df=df.dropna(subset=['report_date','cot_mm_net','cot_open_interest'])
    df=df.sort_values('report_date').drop_duplicates('report_date',keep='last')
    # CONSERVATIVE lag: Tuesday as-of + 10 days (after routine Friday
    # publication, including most holiday shifts); no hypothetical same-Tuesday availability.
    df['available_at']=(df['report_date']+pd.Timedelta(days=10)+pd.Timedelta(hours=23,minutes=59)).dt.tz_localize('UTC')
    out=df[['available_at','report_date','cot_mm_net','cot_open_interest']]
    if len(out)<120:raise RuntimeError('Insufficient CFTC weekly history')
    print('CFTC_ROWS',len(out),'DATE_RANGE',str(out['report_date'].min()),str(out['report_date'].max()))
    return out.reset_index(drop=True)

def official_gld_releases()->pd.DataFrame:
    u="https://api.spdrgoldshares.com/api/v1/historical-archive?exchange=NYSE&lang=en&product=gld"
    workbook=pd.read_excel(io.BytesIO(get(u)),sheet_name="US GLD Historical Archive")
    columns={str(c).strip().lower():c for c in workbook.columns}
    date_col=next((c for k,c in columns.items() if k.startswith('date')),None)
    tonnes_col=next((c for k,c in columns.items() if 'tonnes of gold' in k),None)
    if date_col is None or tonnes_col is None:
        raise ValueError('Official GLD XLSX fields changed '+repr(list(workbook.columns)))
    df=workbook[[date_col,tonnes_col]].rename(columns={date_col:'date',tonnes_col:'etf_tonnes'})
    df['date']=pd.to_datetime(df['date'],errors='coerce')
    df['etf_tonnes']=pd.to_numeric(df['etf_tonnes'],errors='coerce')
    df=df.dropna().sort_values('date').drop_duplicates('date')
    df=df[(df.date>=pd.Timestamp('2023-12-01'))&(df.date<=END)]
    # GLD sponsor reports trust holdings after regular trading hours NY.
    # Next-next calendar day at noon UTC is deliberately conservative.
    df['available_at']=(df['date']+pd.Timedelta(days=2,hours=12)).dt.tz_localize('UTC')
    if len(df)<550:raise RuntimeError('Official GLD history incomplete')
    print('GLD_ROWS',len(df),'DATE_RANGE',str(df.date.min()),str(df.date.max()))
    return df[['available_at','date','etf_tonnes']].reset_index(drop=True)

def eastmoney_contract(code:str)->pd.DataFrame:
    params=dict(EM_PARAMS,secid='113.'+code)
    url="https://push2his.eastmoney.com/api/qt/stock/kline/get?"+urllib.parse.urlencode(params)
    j=get_json(url)
    raw=(j.get('data') or {}).get('klines') or []
    if not raw:raise RuntimeError('Eastmoney no rows for '+code)
    rows=[]
    for s in raw:
        v=s.split(',')
        if len(v)<13:continue
        day=pd.Timestamp(v[0])
        if not BEGIN<=day<=END:continue
        rows.append({'au_date':day,'au_open':float(v[1]),'au_close':float(v[2]),
                     'au_high':float(v[3]),'au_low':float(v[4]),
                     'au_volume':float(v[5]),'au_open_interest':float(v[12]),
                     'au_contract':code})
    df=pd.DataFrame(rows)
    if df.empty:raise RuntimeError('Empty after as-of cutoff for '+code)
    if df.au_date.duplicated().any():raise ValueError('Duplicate AU date')
    if (df.au_close<=0).any() or (df.au_open<=0).any():raise ValueError('Bad AU pricing')
    print('EASTMONEY_AU',code,len(df),str(df.au_date.min()),str(df.au_date.max()),
          'LAST_CLOSE',df.au_close.iloc[-1])
    return df.sort_values('au_date').reset_index(drop=True)

def eastmoney_main_code()->str:
    # Official Eastmoney SHFE mapping: market 113, page 5, 沪金主连 => aum.
    # Avoid repeatedly querying 20 pages (and hitting rate limits).
    candidates=get_json('https://futsse-static.eastmoney.com/redis?msgid=113_5')
    hits=[str(x.get('code')) for x in candidates if x.get('name')=='沪金主连']
    if 'aum' not in hits:
        raise ValueError('Expected 沪金主连/aum mapping absent; halt rather than guess')
    return 'aum'

def make_market(gold:pd.DataFrame,dxy:pd.DataFrame,tips:pd.DataFrame,au:pd.DataFrame,cot:pd.DataFrame)->pd.DataFrame:
    base=gold.rename(columns={'timestamp':'date','close':'gc_close','high':'gc_high','low':'gc_low'})
    base['date']=pd.to_datetime(base['date'])
    base=base[['date','gc_close','gc_high','gc_low','volume']].sort_values('date')
    base=base.merge(dxy[['date','dxy']],on='date',how='left')
    base=base.merge(tips[['date','tips10_pp']],on='date',how='left')
    base['cutoff_at']=pd.to_datetime(base['date'].dt.strftime('%Y-%m-%d')+'T23:59:00Z',utc=True)
    base=pd.merge_asof(base.sort_values('cutoff_at'),
                       cot[['available_at','cot_open_interest']].sort_values('available_at'),
                       left_on='cutoff_at',right_on='available_at',direction='backward')
    # Important: only *previously published WEEKLY* CFTC all-trader OI, not CME daily OI.
    base['comex_oi']=base['cot_open_interest']
    au=au.copy().sort_values('au_date').reset_index(drop=True)
    prev=au.au_date.shift()
    gap=(au.au_date-prev).dt.days
    # Gold night session is 21:00 China time on preceding SHFE business day,
    # represented by au_date; for unusually long holidays, use 09:00 China day.
    start=[]
    for i,row in au.iterrows():
        day=row['au_date']
        if i>0 and 1<=gap.iloc[i]<=3:
            timestamp=prev.iloc[i]+pd.Timedelta(hours=21)
        else:
            timestamp=day+pd.Timedelta(hours=9)
        start.append((timestamp-pd.Timedelta(hours=8)).tz_localize('UTC'))
    au['next_au_open_at']=start
    nxt=pd.merge_asof(base.sort_values('cutoff_at'),
                     au.sort_values('next_au_open_at'),
                     left_on='cutoff_at',right_on='next_au_open_at',
                     direction='forward',allow_exact_matches=False)
    nxt=nxt.dropna(subset=['tips10_pp','dxy','au_open','au_close'])
    # Multiple US data updates in Chinese closed markets map to one actual entry;
    # use only the most recently released snapshot before that AU session.
    nxt=nxt.sort_values('cutoff_at').drop_duplicates('next_au_open_at',keep='last')
    nxt=nxt.sort_values('date').reset_index(drop=True)
    nxt=nxt.rename(columns={'au_open':'au_next_open','au_close':'au_next_close'})
    if (nxt.next_au_open_at<=nxt.cutoff_at).any():raise AssertionError('Leakage entry before signal')
    return nxt[['date','cutoff_at','next_au_open_at','gc_close','gc_high','gc_low',
                'comex_oi','tips10_pp','dxy','au_next_open','au_next_close',
                'au_date','au_contract','au_volume','au_open_interest','cot_open_interest']]

def main():
    gold=pd.read_csv(ROOT/'data/gold_daily.csv')
    gold['timestamp']=pd.to_datetime(gold['timestamp'])
    if len(gold)<700 or gold.timestamp.max().strftime('%Y-%m-%d')!='2026-10-30':
        raise RuntimeError('Gold verified history not updated; abort')
    dxy=yahoo_dxy();tips=fred_tips();cot=cot_releases();etf=official_gld_releases()
    out=ROOT/'data'/'research_inputs'
    out.mkdir(parents=True,exist_ok=True)
    # Keep sponsor proprietary data within ephemeral CI runner; never commit XLSX/GLD raw.
    etf[['available_at','etf_tonnes']].to_csv(out/'gld_releases_LOCAL_ONLY.csv',index=False)
    cot[['available_at','cot_mm_net','cot_open_interest']].to_csv(out/'cftc_releases.csv',index=False)
    main_code=None
    try:
        main_code=eastmoney_main_code()
        au_main=eastmoney_contract(main_code)
        if len(au_main)<500 or au_main.au_date.max()<pd.Timestamp('2026-10-28'):
            raise RuntimeError('AU main recent history incomplete')
        au_main.to_csv(out/'au_main_history.csv',index=False)
        m=make_market(gold,dxy,tips,au_main,cot)
        m.to_csv(out/'gold_v35_au_main_market.csv',index=False)
        print('MAIN_BACKTEST_INPUT',len(m))
    except Exception as e:
        print('AU_MAIN_DATA_BLOCKER',type(e).__name__,str(e)[:300])
    au2612=eastmoney_contract('AU2612')
    au2612.to_csv(out/'au2612_history.csv',index=False)
    m2=make_market(gold,dxy,tips,au2612,cot)
    m2.to_csv(out/'gold_v35_au2612_market.csv',index=False)
    if len(m2)<150 or m2.au_date.max()<pd.Timestamp('2026-10-28'):
        raise RuntimeError('AU2612 insufficient actual tradable history')
    report={'source_gc':'Yahoo GC=F continuous; data/gold_daily.csv',
            'source_dxy':'Yahoo DX-Y.NYB',
            'source_tips':'FRED DFII10, official daily observations',
            'source_cot':'CFTC official disaggregated futures-only 2023-2026 zipped reports',
            'cot_availability':'as-of Tuesday +10 calendar days at 23:59 UTC; CONSERVATIVE, not actual release timestamp',
            'source_gld':'SPDR official GLD historical archive XLSX, analyzed only in CI; not redistributed',
            'gld_availability':'observation date +2 calendar days at 12:00 UTC; conservative',
            'source_au':'Eastmoney SHFE AU2612 fixed contract and (where accessible) main continuous',
            'main_contract_code':main_code,
            'dxy_rows':len(dxy),'fred_rows':len(tips),'cot_reports':len(cot),
            'gld_rows':len(etf),'au2612_rows':len(au2612),'au2612_aligned_rows':len(m2),
            'date_cap':'2026-10-30',
            'open_time_note':'AU session open timestamp reconstructed from preceding exchange trading day; holiday records >3-day gap use 09:00 CN; may require minute-bar audit.',
            'comex_oi_note':'comex_oi is POINT-IN-TIME LAST PUBLISHED WEEKLY CFTC open interest, NOT CME daily OI'}
    (out/'provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('INPUT_REPORT',json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
