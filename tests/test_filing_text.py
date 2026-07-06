from quant_trading.data.filing_text import compute_similarity, strip_filing_html

HIDDEN_XBRL_SAMPLE = """
<html><body>
<div style="display:none">
<ix:header><ix:hidden>
<ix:nonNumeric name="dei:EntityCentralIndexKey">0000320193</ix:nonNumeric>
<ix:nonNumeric name="us-gaap:RevenueRemainingPerformanceObligationExpectedTiming">P3Y</ix:nonNumeric>
</ix:hidden></ix:header>
</div>
<p>The Company designs, manufactures, and markets smartphones.</p>
<script>var trackingCode = "should not appear";</script>
</body></html>
"""


def test_strip_filing_html_removes_hidden_xbrl_metadata():
    text = strip_filing_html(HIDDEN_XBRL_SAMPLE)
    assert "entitycentralindexkey" not in text
    assert "revenueremainingperformanceobligation" not in text
    assert "smartphones" in text


def test_strip_filing_html_removes_scripts():
    text = strip_filing_html(HIDDEN_XBRL_SAMPLE)
    assert "trackingcode" not in text
    assert "should not appear" not in text


def test_strip_filing_html_normalizes_whitespace_and_lowercases():
    html = "<html><body><p>MiXeD   Case\n\n  Text</p></body></html>"
    text = strip_filing_html(html)
    assert text == "mixed case text"


def test_compute_similarity_identical_documents_is_one():
    doc = "the company reported strong revenue growth this quarter across all business segments"
    sim = compute_similarity(doc, doc)
    assert abs(sim - 1.0) < 1e-9


def test_compute_similarity_completely_different_documents_is_low():
    doc_a = "revenue growth margin profit quarterly earnings segment"
    doc_b = "litigation risk factor lawsuit regulatory investigation penalty"
    sim = compute_similarity(doc_a, doc_b)
    assert sim < 0.3


def test_compute_similarity_empty_input_returns_nan():
    import math

    assert math.isnan(compute_similarity("", "some text"))
    assert math.isnan(compute_similarity("some text", ""))
