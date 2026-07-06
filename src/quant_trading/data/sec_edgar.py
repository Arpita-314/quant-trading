"""Async SEC EDGAR client for real, point-in-time insider-transaction data.

Unlike news headlines (only the "current" set is ever retrievable for free),
SEC Form 4 filings carry real historical filing dates going back years, so
a signal built from them can be honestly backtested rather than only used
live. This module fetches filing lists and Form 4 XML documents concurrently
with asyncio, respecting SEC's fair-use rate limit (a bounded semaphore, a
required User-Agent, and an on-disk cache so repeated runs don't re-hit the
network).

Only transaction codes P (open-market purchase) and S (open-market sale) are
kept. Codes like M (option/RSU exercise) and F (tax-withholding disposal)
are administrative, not discretionary, and including them would drown the
real signal (insiders voluntarily buying or selling) in vesting-schedule
noise -- a common mistake in naive insider-trading signals.
"""

from __future__ import annotations

import asyncio
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import pandas as pd

SEC_USER_AGENT = "quant-trading-research contact@example.com"
HEADERS = {"User-Agent": SEC_USER_AGENT}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
MAX_CONCURRENT_REQUESTS = 5

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data_cache" / "sec_edgar"

DISCRETIONARY_CODES = {"P", "S"}  # open-market purchase / sale


@dataclass
class InsiderTransaction:
    ticker: str
    date: str
    code: str  # 'P' or 'S'
    shares: float
    price: float | None

    @property
    def dollar_value(self) -> float:
        return self.shares * (self.price or 0.0)

    @property
    def signed_dollar_value(self) -> float:
        sign = 1.0 if self.code == "P" else -1.0
        return sign * self.dollar_value


REQUEST_TIMEOUT_SECONDS = 60


async def _get_json(session: aiohttp.ClientSession, url: str) -> dict:
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    async with session.get(url, headers=HEADERS, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.json(content_type=None)


async def _get_text(session: aiohttp.ClientSession, url: str) -> str:
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    async with session.get(url, headers=HEADERS, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.text()


_TICKER_TO_CIK_CACHE: dict[str, str] | None = None


async def load_ticker_to_cik(session: aiohttp.ClientSession, tickers: list[str]) -> dict[str, str]:
    """Resolve tickers to zero-padded CIKs. The full ~800KB ticker->CIK file
    is fetched once per process and memoized -- it's the same slow, largely
    static file regardless of which tickers are requested."""
    global _TICKER_TO_CIK_CACHE
    if _TICKER_TO_CIK_CACHE is None:
        data = await _get_json(session, TICKERS_URL)
        _TICKER_TO_CIK_CACHE = {
            entry["ticker"]: str(entry["cik_str"]).zfill(10) for entry in data.values()
        }
    wanted = set(tickers)
    return {t: cik for t, cik in _TICKER_TO_CIK_CACHE.items() if t in wanted}


async def _list_form4_filings(
    session: aiohttp.ClientSession, cik: str, start_date: str | None = None
) -> list[dict]:
    """Return [{filingDate, accessionNumber, primaryDocument}, ...] for Form 4s
    on or after `start_date` (skip entirely, default: no limit).

    Paginated older-filing archives carry their own [filingFrom, filingTo]
    date range in the index -- if that whole range is before `start_date`,
    the page is skipped without fetching it, since none of its filings could
    pass the filter anyway.
    """
    filings: list[dict] = []
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = await _get_json(session, url)
    recent = data["filings"]["recent"]
    for form, date, accn, doc in zip(
        recent["form"], recent["filingDate"], recent["accessionNumber"], recent["primaryDocument"]
    ):
        if form == "4" and (start_date is None or date >= start_date):
            filings.append({"filingDate": date, "accessionNumber": accn, "primaryDocument": doc})

    for older in data["filings"].get("files", []):
        if start_date is not None and older.get("filingTo", "9999-99-99") < start_date:
            continue
        older_data = await _get_json(
            session, f"https://data.sec.gov/submissions/{older['name']}"
        )
        for form, date, accn, doc in zip(
            older_data.get("form", []),
            older_data.get("filingDate", []),
            older_data.get("accessionNumber", []),
            older_data.get("primaryDocument", []),
        ):
            if form == "4" and (start_date is None or date >= start_date):
                filings.append({"filingDate": date, "accessionNumber": accn, "primaryDocument": doc})

    return filings


def _parse_form4_xml(xml_text: str, ticker: str) -> list[InsiderTransaction]:
    root = ET.fromstring(xml_text)
    out = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        code_el = txn.find("./transactionCoding/transactionCode")
        date_el = txn.find("./transactionDate/value")
        shares_el = txn.find("./transactionAmounts/transactionShares/value")
        price_el = txn.find("./transactionAmounts/transactionPricePerShare/value")
        if code_el is None or code_el.text not in DISCRETIONARY_CODES:
            continue
        if date_el is None or shares_el is None:
            continue
        try:
            shares = float(shares_el.text)
        except (TypeError, ValueError):
            continue
        price = None
        if price_el is not None and price_el.text is not None:
            try:
                price = float(price_el.text)
            except ValueError:
                price = None
        out.append(
            InsiderTransaction(
                ticker=ticker, date=date_el.text, code=code_el.text, shares=shares, price=price
            )
        )
    return out


MAX_FETCH_RETRIES = 5


async def _fetch_transactions_for_filing(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    cik: str,
    ticker: str,
    accession: str,
    primary_document: str,
) -> tuple[list[InsiderTransaction], bool]:
    """Returns (transactions, failed). A transient network failure must be
    distinguishable from "this filing genuinely has no P/S transactions" --
    silently treating both as an empty list would make real data gaps look
    like an honest absence of insider activity, corrupting the signal
    without any visible symptom."""
    accn_no_dashes = accession.replace("-", "")
    cik_int = str(int(cik))
    # `primary_document` from the submissions API points to the XSLT-rendered
    # HTML view (e.g. "xslF345X06/form4.xml"); the raw parseable XML always
    # sits at the accession folder root under just the basename.
    doc_basename = primary_document.rsplit("/", 1)[-1]
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_no_dashes}/{doc_basename}"

    async with semaphore:
        xml_text = None
        for attempt in range(MAX_FETCH_RETRIES):
            try:
                xml_text = await _get_text(session, url)
                break
            except Exception:
                if attempt < MAX_FETCH_RETRIES - 1:
                    await asyncio.sleep(min(2.0**attempt, 30.0))
        if xml_text is None:
            return [], True

    try:
        return _parse_form4_xml(xml_text, ticker), False
    except ET.ParseError:
        return [], True


async def fetch_insider_transactions(
    tickers: list[str],
    start_date: str | None = None,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a tidy DataFrame of discretionary insider transactions across `tickers`.

    Columns: ticker, date, code, shares, price, dollar_value, signed_dollar_value.
    `start_date` (e.g. "2019-01-01") skips filings before it entirely, since
    fetching years of filings outside a backtest's price-data window is pure
    waste. Cached to disk per (tickers, start_date) since a full filing
    history can be thousands of HTTP calls across a universe.
    """
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = f"{'-'.join(sorted(tickers))}_{start_date or 'all'}"
        cache_path = cache_dir / f"{key}.parquet"
        if cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        ticker_to_cik = await load_ticker_to_cik(session, tickers)
        missing = [t for t in tickers if t not in ticker_to_cik]
        if missing:
            raise ValueError(f"Could not resolve CIK for: {missing}")

        filing_lists = await asyncio.gather(
            *[_list_form4_filings(session, ticker_to_cik[t], start_date=start_date) for t in tickers]
        )

        jobs = []  # (ticker, accession, primary_document) for every filing across every ticker
        for ticker, filings in zip(tickers, filing_lists):
            for f in filings:
                jobs.append((ticker, f["accessionNumber"], f["primaryDocument"]))

        all_txns: list = []
        remaining = jobs
        # A correlated network outage can outlast one filing's own retry loop;
        # re-sweep whatever is still failing a few times, with a real pause
        # between sweeps, rather than trusting a single pass.
        for sweep in range(3):
            if not remaining:
                break
            if sweep > 0:
                print(f"[sec_edgar] retry sweep {sweep}: re-fetching {len(remaining)} failed filings...")
                await asyncio.sleep(15.0)
            results = await asyncio.gather(
                *[
                    _fetch_transactions_for_filing(
                        session, semaphore, ticker_to_cik[ticker], ticker, accn, doc
                    )
                    for ticker, accn, doc in remaining
                ]
            )
            still_failing = []
            for job, (txns, failed) in zip(remaining, results):
                if failed:
                    still_failing.append(job)
                else:
                    all_txns.extend(txns)
            remaining = still_failing

    if remaining:
        print(
            f"[sec_edgar] WARNING: {len(remaining)}/{len(jobs)} filings failed to fetch/parse "
            "after all retries -- the resulting signal has real gaps, not confirmed zero activity."
        )
    rows = [
        {
            "ticker": t.ticker,
            "date": t.date,
            "code": t.code,
            "shares": t.shares,
            "price": t.price,
            "dollar_value": t.dollar_value,
            "signed_dollar_value": t.signed_dollar_value,
        }
        for t in all_txns
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    if cache_path is not None:
        df.to_parquet(cache_path)

    return df


def build_daily_insider_flow(transactions: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Pivot tidy transactions into a wide daily net signed-dollar-flow matrix
    aligned to `price_index`, filling non-filing days with 0 (no news that day,
    not "flat position" -- the strategy decides how to turn flow into a position)."""
    tickers = sorted(transactions["ticker"].unique()) if not transactions.empty else []
    daily = pd.DataFrame(0.0, index=price_index, columns=tickers)
    if transactions.empty:
        return daily
    grouped = transactions.groupby(["date", "ticker"])["signed_dollar_value"].sum()
    for (date, ticker), value in grouped.items():
        if date in daily.index and ticker in daily.columns:
            daily.loc[date, ticker] += value
    return daily
