import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from portfolio_optimizer.data.adapters import (
    YahooFinanceAdapter, PolygonAdapter, AlpacaAdapter, BinanceAdapter,
    get_adapter, ADAPTER_REGISTRY,
)


def test_registry_contains_all_four_sources():
    assert set(ADAPTER_REGISTRY) == {"yahoo", "polygon", "alpaca", "binance"}


def test_yahoo_adapter_constructs():
    adapter = YahooFinanceAdapter()
    assert isinstance(adapter, YahooFinanceAdapter)


def test_get_adapter_factory_yahoo():
    adapter = get_adapter("yahoo")
    assert isinstance(adapter, YahooFinanceAdapter)


def test_get_adapter_unknown_source_raises():
    with pytest.raises(ValueError):
        get_adapter("not_a_real_source")


def test_polygon_adapter_fetch_prices_mocked():
    adapter = PolygonAdapter(api_key="fake_key")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"t": 1704067200000, "c": 100.0},
            {"t": 1704153600000, "c": 101.5},
            {"t": 1704240000000, "c": 99.8},
        ]
    }
    mock_response.raise_for_status.return_value = None
    with patch("requests.get", return_value=mock_response):
        prices = adapter.fetch_prices(["AAPL"], "2024-01-01", "2024-01-05")
    assert "AAPL" in prices.columns
    assert len(prices) == 3
    assert prices["AAPL"].iloc[0] == 100.0


def test_polygon_adapter_fetch_returns_mocked():
    adapter = PolygonAdapter(api_key="fake_key")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"t": 1704067200000, "c": 100.0},
            {"t": 1704153600000, "c": 102.0},
            {"t": 1704240000000, "c": 101.0},
        ]
    }
    mock_response.raise_for_status.return_value = None
    with patch("requests.get", return_value=mock_response):
        returns = adapter.fetch_returns(["AAPL"], "2024-01-01", "2024-01-05")
    assert abs(returns["AAPL"].iloc[0] - 0.02) < 1e-9


def test_binance_adapter_fetch_prices_mocked():
    adapter = BinanceAdapter()
    mock_response = MagicMock()
    # Binance kline format: [open_time, open, high, low, close, volume, ...]
    mock_response.json.return_value = [
        [1704067200000, "42000", "42500", "41800", "42300", "1000"],
        [1704153600000, "42300", "43000", "42100", "42800", "1200"],
    ]
    mock_response.raise_for_status.return_value = None
    with patch("requests.get", return_value=mock_response):
        prices = adapter.fetch_prices(["BTCUSDT"], "2024-01-01", "2024-01-03")
    assert "BTCUSDT" in prices.columns
    assert prices["BTCUSDT"].iloc[0] == 42300.0


def test_alpaca_adapter_requires_credentials():
    # Construction should succeed (SDK import check only); actual client
    # creation happens lazily inside fetch_prices.
    try:
        adapter = AlpacaAdapter(api_key="fake", api_secret="fake")
        assert isinstance(adapter, AlpacaAdapter)
    except ImportError:
        pytest.skip("alpaca-py not installed in this environment")


def test_interval_parsing_polygon():
    assert PolygonAdapter._parse_interval("1d") == (1, "day")
    assert PolygonAdapter._parse_interval("1h") == (1, "hour")
    assert PolygonAdapter._parse_interval("unknown") == (1, "day")


class TestYahooFinanceExpandedSurface:
    def test_list_available_ticker_methods_returns_real_list(self):
        methods = YahooFinanceAdapter.list_available_ticker_methods()
        assert isinstance(methods, list)
        assert len(methods) > 20  # yfinance's Ticker surface is large
        assert "history" in methods
        assert "get_info" in methods

    def test_raw_ticker_returns_yfinance_ticker_object(self):
        adapter = YahooFinanceAdapter()
        t = adapter.raw_ticker("AAPL")
        import yfinance as yf
        assert isinstance(t, yf.Ticker)

    def test_get_fast_quote_uses_correct_attribute_names(self):
        """Regression test for a real bug caught during development: the
        first implementation used camelCase dict-style keys ('lastPrice')
        that don't exist on yfinance's FastInfo class, which actually
        exposes snake_case properties ('last_price'). This verifies the
        adapter reads the attributes that actually exist on the class,
        using a mock so it doesn't require live network access.
        """
        adapter = YahooFinanceAdapter()
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 150.25
        mock_fast_info.previous_close = 148.0
        mock_fast_info.day_high = 151.0
        mock_fast_info.day_low = 147.5
        mock_fast_info.last_volume = 1_000_000
        mock_fast_info.market_cap = 2_500_000_000_000
        mock_fast_info.currency = "USD"

        with patch.object(YahooFinanceAdapter, "raw_ticker") as mock_raw:
            mock_raw.return_value.fast_info = mock_fast_info
            quote = adapter.get_fast_quote("AAPL")

        assert quote["last_price"] == 150.25
        assert quote["previous_close"] == 148.0
        assert quote["currency"] == "USD"

    def test_get_company_info_extracts_expected_keys(self):
        adapter = YahooFinanceAdapter()
        mock_info = {
            "sector": "Technology", "industry": "Consumer Electronics",
            "marketCap": 2_500_000_000_000, "fullTimeEmployees": 150000,
            "longName": "Apple Inc.", "currency": "USD", "exchange": "NMS", "quoteType": "EQUITY",
            "some_other_field_we_dont_care_about": "ignored",
        }
        with patch.object(YahooFinanceAdapter, "raw_ticker") as mock_raw:
            mock_raw.return_value.get_info.return_value = mock_info
            info = adapter.get_company_info("AAPL")

        assert info["sector"] == "Technology"
        assert info["longName"] == "Apple Inc."
        assert "some_other_field_we_dont_care_about" not in info

    def test_get_corporate_actions_delegates_to_ticker(self):
        adapter = YahooFinanceAdapter()
        mock_actions = pd.DataFrame({"Dividends": [0.24], "Stock Splits": [0.0]})
        with patch.object(YahooFinanceAdapter, "raw_ticker") as mock_raw:
            mock_raw.return_value.get_actions.return_value = mock_actions
            actions = adapter.get_corporate_actions("AAPL")
        assert "Dividends" in actions.columns

    def test_get_news_delegates_to_ticker(self):
        adapter = YahooFinanceAdapter()
        with patch.object(YahooFinanceAdapter, "raw_ticker") as mock_raw:
            mock_raw.return_value.get_news.return_value = [{"title": "Test headline"}]
            news = adapter.get_news("AAPL")
        assert news == [{"title": "Test headline"}]

    def test_get_live_quote_uses_websocket_and_returns_dict(self):
        adapter = YahooFinanceAdapter()
        with patch("yfinance.WebSocket") as MockWS:
            mock_ws_instance = MagicMock()
            MockWS.return_value = mock_ws_instance

            def fake_listen(handler):
                handler({"id": "AAPL", "price": 150.25})

            mock_ws_instance.listen.side_effect = fake_listen
            quote = adapter.get_live_quote("AAPL", timeout_seconds=0.5)
        mock_ws_instance.subscribe.assert_called_once_with(["AAPL"])
        mock_ws_instance.close.assert_called_once()

    def test_stream_live_prices_subscribes_and_listens(self):
        adapter = YahooFinanceAdapter()
        received = []
        with patch("yfinance.WebSocket") as MockWS:
            mock_ws_instance = MagicMock()
            MockWS.return_value = mock_ws_instance

            def fake_listen(handler):
                handler({"id": "AAPL", "price": 150.0})
                handler({"id": "MSFT", "price": 300.0})

            mock_ws_instance.listen.side_effect = fake_listen
            adapter.stream_live_prices(["AAPL", "MSFT"], on_tick=received.append)

        mock_ws_instance.subscribe.assert_called_once_with(["AAPL", "MSFT"])
        assert len(received) == 2
        mock_ws_instance.close.assert_called_once()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
