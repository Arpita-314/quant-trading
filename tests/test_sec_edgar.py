import pandas as pd

from quant_trading.data.sec_edgar import _parse_form4_xml, build_daily_insider_flow

SAMPLE_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionDate><value>2024-03-04</value></transactionDate>
            <transactionCoding>
                <transactionCode>P</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>1000</value></transactionShares>
                <transactionPricePerShare><value>50.25</value></transactionPricePerShare>
            </transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <transactionDate><value>2024-03-04</value></transactionDate>
            <transactionCoding>
                <transactionCode>S</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>400</value></transactionShares>
                <transactionPricePerShare><value>51.00</value></transactionPricePerShare>
            </transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <transactionDate><value>2024-03-04</value></transactionDate>
            <transactionCoding>
                <transactionCode>M</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>5000</value></transactionShares>
                <transactionPricePerShare><footnoteId id="F1"/></transactionPricePerShare>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"""


def test_parse_form4_keeps_only_discretionary_codes():
    txns = _parse_form4_xml(SAMPLE_FORM4_XML, ticker="ZZZ")
    codes = {t.code for t in txns}
    assert codes == {"P", "S"}  # the M (exercise) transaction must be dropped


def test_parse_form4_signed_dollar_value():
    txns = _parse_form4_xml(SAMPLE_FORM4_XML, ticker="ZZZ")
    by_code = {t.code: t for t in txns}
    assert by_code["P"].signed_dollar_value == 1000 * 50.25
    assert by_code["S"].signed_dollar_value == -(400 * 51.00)


def test_parse_form4_handles_missing_price_gracefully():
    # a transaction with no transactionPricePerShare value (footnote-only) is
    # still kept if it has a discretionary code, with price=None
    xml = SAMPLE_FORM4_XML.replace(
        "<transactionCode>M</transactionCode>", "<transactionCode>P</transactionCode>"
    )
    txns = _parse_form4_xml(xml, ticker="ZZZ")
    no_price = [t for t in txns if t.price is None]
    assert len(no_price) == 1
    assert no_price[0].dollar_value == 0.0


def test_build_daily_insider_flow_pivots_and_aligns_to_price_index():
    transactions = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "date": pd.to_datetime(["2024-03-04", "2024-03-04", "2024-03-05"]),
            "signed_dollar_value": [50250.0, -20400.0, 1000.0],
        }
    )
    price_index = pd.bdate_range("2024-03-01", periods=10)
    flow = build_daily_insider_flow(transactions, price_index)

    assert list(flow.index) == list(price_index)
    assert set(flow.columns) == {"AAA", "BBB"}
    assert flow.loc[pd.Timestamp("2024-03-04"), "AAA"] == 50250.0 - 20400.0
    assert flow.loc[pd.Timestamp("2024-03-05"), "BBB"] == 1000.0
    assert flow.loc[pd.Timestamp("2024-03-01"), "AAA"] == 0.0


def test_build_daily_insider_flow_empty_transactions():
    price_index = pd.bdate_range("2024-03-01", periods=5)
    flow = build_daily_insider_flow(pd.DataFrame(), price_index)
    assert flow.shape == (5, 0)
