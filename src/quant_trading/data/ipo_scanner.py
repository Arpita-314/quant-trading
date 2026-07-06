"""Recent-IPO scanner via SEC EDGAR full text search.

Finds companies that recently filed a Form 424B4 (the final IPO prospectus
-- filed when a deal actually prices and the stock starts trading) rather
than Form S-1, which only signals *intent* to go public and includes many
deals that later get withdrawn or delayed indefinitely. This answers "what's
newly public" on demand instead of a static, quickly-stale hardcoded ticker
list.

SPAC/blank-check shells file 424B4s too, when the empty shell itself IPOs
before acquiring an operating business -- excluded by default via SIC code
6770, since "recent startup IPOs" means operating companies, not blank-check
vehicles waiting for a merger target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import aiohttp

FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
HEADERS = {"User-Agent": "quant-trading-research contact@example.com"}
BLANK_CHECK_SICS = {"6770"}
PAGE_SIZE = 100


@dataclass
class IpoFiling:
    ticker: str | None
    company_name: str
    cik: str
    filing_date: str
    sic: str | None


def _extract_ticker(display_name: str) -> str | None:
    """display_names look like 'Cerebras Systems Inc.  (CBRS)  (CIK 0002021728)',
    but a company with no assigned ticker yet renders as just
    'DSC Holdings Ltd.  (CIK 0001966041)' -- the trailing CIK group is always
    present, so naively taking the first parenthesized group misreads the
    CIK itself as a ticker whenever the ticker group is absent."""
    groups = re.findall(r"\(([^)]*)\)", display_name)
    for group in groups:
        if group.startswith("CIK "):
            continue
        ticker = group.split(",")[0].strip()  # drop warrant/unit variants after the primary ticker
        return ticker or None
    return None


async def fetch_recent_ipos(
    start_date: str,
    end_date: str,
    exclude_spacs: bool = True,
    max_results: int = 500,
) -> list[IpoFiling]:
    """Return de-duplicated (by CIK) 424B4 filers between start_date and end_date."""
    results: list[IpoFiling] = []
    seen_ciks: set[str] = set()

    async with aiohttp.ClientSession() as session:
        offset = 0
        while offset < max_results:
            url = (
                f"{FULL_TEXT_SEARCH_URL}?forms=424B4&dateRange=custom"
                f"&startdt={start_date}&enddt={end_date}&from={offset}"
            )
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                data = await resp.json()

            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break

            for h in hits:
                src = h["_source"]
                ciks = src.get("ciks") or []
                if not ciks or ciks[0] in seen_ciks:
                    continue
                sics = src.get("sics") or []
                if exclude_spacs and any(s in BLANK_CHECK_SICS for s in sics):
                    continue
                seen_ciks.add(ciks[0])
                name = src["display_names"][0]
                results.append(
                    IpoFiling(
                        ticker=_extract_ticker(name),
                        company_name=name.split("(")[0].strip(),
                        cik=ciks[0],
                        filing_date=src["file_date"],
                        sic=sics[0] if sics else None,
                    )
                )

            total = data.get("hits", {}).get("total", {}).get("value", 0)
            offset += len(hits)
            if offset >= total:
                break

    return results
