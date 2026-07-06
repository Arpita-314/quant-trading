"""10-K/10-Q filing text and filing-to-filing similarity ("Lazy Prices").

Implements the measurement at the core of Cohen, Malloy & Nguyen, "Lazy
Prices" (Journal of Finance, 2020): the cosine similarity between a
company's current annual/quarterly filing and its immediately preceding
same-type filing. Unusually large language changes (low similarity) predict
negative subsequent returns -- the paper's argument is that markets
underreact to lengthy, boring textual disclosures that few investors read
closely. Cosine similarity on a term-frequency vector space is literally
the measure used in the paper, not a simplified stand-in for embeddings.

Unlike news headlines, 10-K/10-Q filing dates are real historical
timestamps going back years, so a signal built from this is honestly
backtestable rather than live-report-only.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .sec_edgar import _get_text, _list_filings, load_ticker_to_cik

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data_cache" / "filing_text"
MAX_CONCURRENT_REQUESTS = 5
MAX_FETCH_RETRIES = 5
MAX_TEXT_CHARS = 400_000  # generous cap; covers Item 1/1A/7 fully for essentially all real 10-Ks

_HIDDEN_STYLE_RE = re.compile(r"display\s*:\s*none")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class FilingRecord:
    ticker: str
    filing_date: str
    accession: str
    text: str


def strip_filing_html(raw_html: str) -> str:
    """Extract visible prose from a 10-K/10-Q HTML document.

    Modern filings use inline XBRL: financial facts are tagged directly in
    the HTML, with large blocks of pure metadata hidden via
    `style="display:none"`. BeautifulSoup's plain get_text() doesn't know
    about CSS visibility, so it happily returns that hidden metadata as if
    it were prose -- silently drowning the genuine language-change signal
    in inline-XBRL tag noise. Hidden elements (and <script>/<style>) are
    removed before extracting text.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup.find_all(style=_HIDDEN_STYLE_RE):
        tag.decompose()
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return text[:MAX_TEXT_CHARS]


def compute_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two documents' TF-IDF vectors -- the exact
    measure used in Cohen, Malloy & Nguyen (2020), not a stand-in for it."""
    if not text_a or not text_b:
        return float("nan")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10_000)
    try:
        tfidf = vectorizer.fit_transform([text_a, text_b])
    except ValueError:
        return float("nan")  # e.g. both documents are entirely stopwords/empty after filtering
    return float(cosine_similarity(tfidf[0], tfidf[1])[0, 0])


async def _fetch_filing_text(
    session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, cik: str, accession: str, primary_document: str
) -> str | None:
    accn_no_dashes = accession.replace("-", "")
    cik_int = str(int(cik))
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_no_dashes}/{primary_document}"

    async with semaphore:
        for attempt in range(MAX_FETCH_RETRIES):
            try:
                raw_html = await _get_text(session, url)
                return strip_filing_html(raw_html)
            except Exception:
                if attempt < MAX_FETCH_RETRIES - 1:
                    await asyncio.sleep(min(2.0**attempt, 30.0))
    return None


async def fetch_filing_drift(
    tickers: list[str],
    form_type: str = "10-K",
    start_date: str | None = None,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a tidy DataFrame: ticker, filing_date, similarity (cosine
    similarity to the immediately preceding same-type filing for that
    ticker; NaN for a ticker's very first filing, which has no predecessor).
    """
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = f"{'-'.join(sorted(tickers))}_{form_type}_{start_date or 'all'}"
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
            *[_list_filings(session, ticker_to_cik[t], form_type, start_date=start_date) for t in tickers]
        )
        for filings in filing_lists:
            filings.sort(key=lambda f: f["filingDate"])

        # (ticker, filing) jobs across every ticker, fetched together rather
        # than one ticker fully at a time -- a transient rate-limit window
        # hitting whichever ticker happens to go first shouldn't be able to
        # silently blank out just that one ticker's entire history while
        # everyone processed later on the far side of the window succeeds.
        jobs = [(ticker, filing) for ticker, filings in zip(tickers, filing_lists) for filing in filings]
        texts: dict[tuple[str, str], str | None] = {}
        remaining = jobs
        for sweep in range(3):
            if not remaining:
                break
            if sweep > 0:
                print(f"[filing_text] retry sweep {sweep}: re-fetching {len(remaining)} filings...")
                await asyncio.sleep(15.0)
            cik_by_ticker = ticker_to_cik
            results = await asyncio.gather(
                *[
                    _fetch_filing_text(
                        session, semaphore, cik_by_ticker[ticker], filing["accessionNumber"], filing["primaryDocument"]
                    )
                    for ticker, filing in remaining
                ]
            )
            still_failing = []
            for (ticker, filing), text in zip(remaining, results):
                if text is None:
                    still_failing.append((ticker, filing))
                else:
                    texts[(ticker, filing["accessionNumber"])] = text
            remaining = still_failing

    if remaining:
        print(
            f"[filing_text] WARNING: {len(remaining)}/{len(jobs)} filings failed to fetch "
            "after all retries -- their similarity score is a real gap, not a genuine absence of change."
        )

    rows = []
    for ticker, filings in zip(tickers, filing_lists):
        prev_text = None
        for filing in filings:
            text = texts.get((ticker, filing["accessionNumber"]))
            similarity = compute_similarity(prev_text, text) if prev_text is not None and text is not None else float("nan")
            rows.append({"ticker": ticker, "filing_date": filing["filingDate"], "similarity": similarity})
            if text:
                prev_text = text

    df = pd.DataFrame(rows)
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

    if cache_path is not None:
        df.to_parquet(cache_path)

    return df
