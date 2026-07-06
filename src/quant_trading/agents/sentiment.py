"""Pluggable sentiment scoring for scraped headlines.

Two implementations behind one interface:

- `LexiconSentimentAnalyzer` (default): deterministic, free, no API key,
  same score every time for the same input. Used whenever no key is
  configured, so tests and the live-report script always work out of the
  box.
- `LLMSentimentAnalyzer` (optional): an actual model reads the headlines and
  makes a qualitative bullish/bearish call with a one-line rationale --
  useful for messy, context-dependent headlines a keyword lexicon misreads,
  at the cost of an API key, latency, and non-determinism. Only activates
  if `ANTHROPIC_API_KEY` is set.

Neither is used inside a backtest: sentiment here only ever runs against
*current* headlines (see `data/news_scraper.py` for why), so it's wired into
the live report path only, never into historical strategy scoring.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class SentimentAnalyzer(ABC):
    @abstractmethod
    async def score(self, headlines: list[str]) -> tuple[float, str]:
        """Return (sentiment in [-1, 1], one-line rationale)."""
        raise NotImplementedError


class LexiconSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    async def score(self, headlines: list[str]) -> tuple[float, str]:
        if not headlines:
            return 0.0, "no headlines available"
        compounds = [self._analyzer.polarity_scores(h)["compound"] for h in headlines]
        avg = sum(compounds) / len(compounds)
        return avg, f"VADER compound average over {len(headlines)} headlines"


class LLMSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self, model: str = "claude-sonnet-5"):
        import anthropic

        self._client = anthropic.AsyncAnthropic()
        self._model = model

    async def score(self, headlines: list[str]) -> tuple[float, str]:
        if not headlines:
            return 0.0, "no headlines available"
        joined = "\n".join(f"- {h}" for h in headlines[:15])
        prompt = (
            "Rate the net sentiment of these financial news headlines for the "
            "underlying stock, from -1 (very bearish) to 1 (very bullish). "
            "Respond with exactly two lines: the numeric score, then a "
            "one-sentence rationale.\n\n" + joined
        )
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        lines = text.splitlines()
        try:
            score = max(-1.0, min(1.0, float(lines[0].strip())))
        except (ValueError, IndexError):
            score = 0.0
        rationale = lines[1].strip() if len(lines) > 1 else text
        return score, rationale


def default_sentiment_analyzer() -> SentimentAnalyzer:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return LLMSentimentAnalyzer()
        except Exception:
            pass
    return LexiconSentimentAnalyzer()
