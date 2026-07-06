from quant_trading.data.ipo_scanner import _extract_ticker


def test_extract_ticker_normal_case():
    assert _extract_ticker("Cerebras Systems Inc.  (CBRS)  (CIK 0002021728)") == "CBRS"


def test_extract_ticker_drops_warrant_unit_suffixes():
    assert _extract_ticker("Idaho Copper Corp  (COPR, COPR-WT)  (CIK 0001263364)") == "COPR"


def test_extract_ticker_missing_ticker_returns_none():
    """A company with no ticker assigned yet renders as just the CIK group --
    naively taking the first parenthesized group would misread the CIK
    itself as a ticker. This is a regression test for that exact bug."""
    assert _extract_ticker("DSC Holdings Ltd.  (CIK 0001966041)") is None


def test_extract_ticker_no_parens_returns_none():
    assert _extract_ticker("No Parens At All") is None
