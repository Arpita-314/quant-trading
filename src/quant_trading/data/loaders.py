"""Market data loading with an on-disk cache.

All strategies in this repo consume a single wide DataFrame of daily close
prices (index = date, columns = ticker). This module is the only place that
talks to an external data source, so swapping yfinance for a broker feed or
a vendor API later means touching one file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data_cache"


def load_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    interval: str = "1d",
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download adjusted close prices for `tickers` between `start` and `end`.

    Results are cached to `cache_dir` as parquet, keyed by tickers/dates, so
    repeated backtest runs during development don't re-hit the network.
    """
    tickers = sorted(set(tickers))
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = f"{'-'.join(tickers)}_{start}_{end or 'latest'}_{interval}".replace("/", "_")
        cache_path = cache_dir / f"{key}.parquet"
        if cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {tickers} in [{start}, {end}]")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.reindex(columns=tickers)
    missing = [t for t in tickers if prices[t].dropna().empty]
    if missing:
        raise ValueError(f"No price data returned for: {missing} (check tickers/date range)")

    # Only drop rows where EVERY ticker is still NaN (before any of them
    # listed). A recently-IPO'd ticker mixed into an older universe legally
    # has NaN for its pre-IPO history -- `how="any"` would silently squeeze
    # the whole frame down to just the newest ticker's short window, wiping
    # out years of history for every other ticker. Downstream strategies
    # already treat per-column NaN correctly (no signal until enough data
    # exists), so there's no need to force row-wise completeness here.
    prices = prices.ffill().dropna(how="all")

    if cache_path is not None:
        prices.to_parquet(cache_path)

    return prices
