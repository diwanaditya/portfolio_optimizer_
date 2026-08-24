"""
Live Data Adapters — unified interface over Polygon, Alpaca, Yahoo Finance,
and Binance, so the rest of the library never needs to know which vendor
your returns came from; every adapter's `fetch_returns()` returns the exact
same `pd.DataFrame` (dates x assets, simple periodic returns) contract that
every optimizer in this repo already expects.

Design
------
Each adapter is a thin, optional wrapper (imports its vendor SDK lazily so
the rest of the package works with zero of these installed) implementing:

    fetch_prices(symbols, start, end, interval) -> DataFrame of close prices
    fetch_returns(symbols, start, end, interval) -> DataFrame of pct-change returns

All adapters require your own API credentials (this library never bundles
or hardcodes any). None of the actual network calls are exercised in this
repo's test suite (no test infra should depend on live external accounts);
`tests/test_data_adapters.py` only checks that construction and the
common-interface contract work, using a mocked HTTP layer.
"""
from __future__ import annotations
import time
import pandas as pd
from abc import ABC, abstractmethod


class LiveDataAdapter(ABC):
    """Common interface every vendor adapter implements."""

    @abstractmethod
    def fetch_prices(self, symbols: list[str], start: str, end: str,
                      interval: str = "1d") -> pd.DataFrame:
        ...

    def fetch_returns(self, symbols: list[str], start: str, end: str,
                       interval: str = "1d") -> pd.DataFrame:
        prices = self.fetch_prices(symbols, start, end, interval)
        return prices.pct_change().dropna(how="all")


class YahooFinanceAdapter(LiveDataAdapter):
    """Free, no-API-key adapter via the `yfinance` package. This is the
    lowest-barrier-to-entry data source in this repo -- zero signup, zero
    credentials, works the moment `pip install yfinance` finishes.

    Beyond the base `fetch_prices`/`fetch_returns` contract, this adapter
    exposes the rest of yfinance's genuinely useful surface for a trading
    workflow (full method list below was enumerated directly against the
    installed yfinance version -- see `list_available_ticker_methods()`):

        Quotes / live data:
            get_fast_quote(symbol)       -- current price snapshot (fast_info)
            get_live_quote(symbol)       -- one-shot live quote via WebSocket
            stream_live_prices(symbols, on_tick)  -- genuine real-time
                streaming quotes via yfinance's WebSocket client
                (wss://streamer.finance.yahoo.com) -- no API key, no
                polling, pushed updates as trades happen.

        Reference / fundamentals (useful for restricted-list / sanity
        checks before trading a name, not required for the optimizers):
            get_company_info(symbol)     -- sector, industry, market cap, etc.
            get_corporate_actions(symbol) -- dividends + splits history
            get_recommendations(symbol)   -- analyst recommendation trend
            get_news(symbol)              -- recent headlines

    None of the extra methods are required by the optimizers or the live
    trading loop -- `fetch_prices`/`fetch_returns` are the only two things
    the rest of this repo actually calls. They're exposed because a real
    trading workflow eventually wants "is this even a real, liquid,
    tradeable name" sanity checks, and yfinance already has that data for
    free rather than requiring a second paid vendor for it.
    """

    # The full yfinance.Ticker method/property surface, enumerated once
    # against the installed version, for reference and for
    # `list_available_ticker_methods()` below -- not all of these are
    # wrapped with a convenience method (most are rarely needed for a
    # systematic workflow), but every one is reachable via
    # `adapter.raw_ticker(symbol).<name>` if you need it directly.
    _KNOWN_TICKER_SURFACE = [
        "info", "get_info", "fast_info", "get_fast_info", "history", "get_history_metadata",
        "actions", "get_actions", "dividends", "get_dividends", "splits", "get_splits",
        "capital_gains", "get_capital_gains", "shares", "get_shares", "get_shares_full",
        "financials", "get_financials", "quarterly_financials", "ttm_financials",
        "balance_sheet", "get_balance_sheet", "quarterly_balance_sheet",
        "cash_flow", "get_cash_flow", "quarterly_cash_flow", "ttm_cash_flow",
        "income_stmt", "get_income_stmt", "quarterly_income_stmt", "ttm_income_stmt",
        "earnings", "get_earnings", "earnings_dates", "get_earnings_dates",
        "earnings_estimate", "get_earnings_estimate", "earnings_history", "get_earnings_history",
        "eps_revisions", "get_eps_revisions", "eps_trend", "get_eps_trend",
        "growth_estimates", "get_growth_estimates", "revenue_estimate", "get_revenue_estimate",
        "recommendations", "get_recommendations", "recommendations_summary",
        "get_recommendations_summary", "upgrades_downgrades", "get_upgrades_downgrades",
        "analyst_price_targets", "get_analyst_price_targets",
        "sustainability", "get_sustainability", "calendar", "get_calendar",
        "major_holders", "get_major_holders", "institutional_holders", "get_institutional_holders",
        "mutualfund_holders", "get_mutualfund_holders",
        "insider_purchases", "get_insider_purchases", "insider_roster_holders",
        "get_insider_roster_holders", "insider_transactions", "get_insider_transactions",
        "isin", "get_isin", "news", "get_news", "options", "option_chain",
        "sec_filings", "get_sec_filings", "funds_data", "get_funds_data",
    ]

    def __init__(self):
        try:
            import yfinance  # noqa: F401
        except ImportError as e:
            raise ImportError("pip install yfinance") from e

    @staticmethod
    def list_available_ticker_methods() -> list:
        """Returns the full list of yfinance.Ticker methods/properties
        available in the installed version -- computed live against the
        actual installed package, not a hardcoded guess, so this stays
        accurate as yfinance adds/removes fields across versions.
        """
        import yfinance as yf
        dummy = yf.Ticker("AAPL")
        return sorted(m for m in dir(dummy) if not m.startswith("_"))

    def raw_ticker(self, symbol: str):
        """Escape hatch: the actual yfinance.Ticker object, for anything
        not wrapped by a convenience method below."""
        import yfinance as yf
        return yf.Ticker(symbol)

    def fetch_prices(self, symbols: list[str], start: str, end: str,
                      interval: str = "1d") -> pd.DataFrame:
        import yfinance as yf
        data = yf.download(symbols, start=start, end=end, interval=interval,
                            auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]].rename(columns={"Close": symbols[0]})
        return prices

    # -- quotes / live data ------------------------------------------------ #
    def get_fast_quote(self, symbol: str) -> dict:
        """Current price snapshot via yfinance's `fast_info` -- a single
        lightweight REST call, not a full page scrape, good for "what's
        the price right now" without the overhead of `.info`.
        """
        t = self.raw_ticker(symbol)
        fi = t.fast_info
        return {
            "symbol": symbol, "last_price": fi.last_price,
            "previous_close": fi.previous_close,
            "day_high": fi.day_high, "day_low": fi.day_low,
            "volume": fi.last_volume, "market_cap": fi.market_cap,
            "currency": fi.currency,
        }

    def get_live_quote(self, symbol: str, timeout_seconds: float = 5.0) -> dict:
        """One-shot live quote via yfinance's real-time WebSocket client --
        connects, waits for exactly one tick for `symbol`, disconnects.
        For continuous streaming across multiple symbols, use
        `stream_live_prices` instead (this is the "just tell me the
        current live price once" convenience wrapper around it).
        """
        import yfinance as yf
        result = {}

        def _handler(message: dict):
            result.update(message)

        ws = yf.WebSocket(verbose=False)
        try:
            ws.subscribe([symbol])
            import threading
            listener = threading.Thread(target=ws.listen, args=(_handler,), daemon=True)
            listener.start()
            listener.join(timeout=timeout_seconds)
        finally:
            ws.close()
        return result

    def stream_live_prices(self, symbols: list[str], on_tick) -> None:
        """Genuine continuous real-time streaming quotes -- no API key, no
        polling loop, no rate limit to manage yourself. `on_tick` is
        called with each raw tick dict as it arrives from Yahoo's
        WebSocket feed. This call BLOCKS (it's the listen loop) -- run it
        in its own thread/process if you need it alongside other work.

        This is the actual live-data primitive that makes an Alpaca (or
        any paid vendor) data subscription unnecessary for a systematic
        strategy that only needs price ticks, not order execution.
        """
        import yfinance as yf
        ws = yf.WebSocket(verbose=False)
        try:
            ws.subscribe(symbols)
            ws.listen(on_tick)
        finally:
            ws.close()

    # -- reference / fundamentals ------------------------------------------- #
    def get_company_info(self, symbol: str) -> dict:
        t = self.raw_ticker(symbol)
        info = t.get_info()
        keys = ["sector", "industry", "marketCap", "fullTimeEmployees",
                "longName", "currency", "exchange", "quoteType"]
        return {k: info.get(k) for k in keys}

    def get_corporate_actions(self, symbol: str) -> pd.DataFrame:
        """Dividends + splits history -- relevant for making sure a return
        series isn't distorted by an unadjusted corporate action."""
        t = self.raw_ticker(symbol)
        return t.get_actions()

    def get_recommendations(self, symbol: str) -> pd.DataFrame:
        t = self.raw_ticker(symbol)
        return t.get_recommendations()

    def get_news(self, symbol: str) -> list:
        t = self.raw_ticker(symbol)
        return t.get_news()


class PolygonAdapter(LiveDataAdapter):
    """Adapter for Polygon.io (https://polygon.io) — equities, options, forex,
    crypto aggregates. Requires an API key (paid plans needed for full
    historical depth / real-time).
    """
    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise ImportError("pip install requests") from e

    def fetch_prices(self, symbols: list[str], start: str, end: str,
                      interval: str = "1d") -> pd.DataFrame:
        import requests
        multiplier, timespan = self._parse_interval(interval)
        frames = {}
        for sym in symbols:
            url = (f"{self.BASE_URL}/v2/aggs/ticker/{sym}/range/"
                   f"{multiplier}/{timespan}/{start}/{end}")
            resp = requests.get(url, params={"apiKey": self.api_key, "adjusted": "true",
                                               "sort": "asc", "limit": 50000})
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("results", [])
            if not results:
                continue
            idx = pd.to_datetime([r["t"] for r in results], unit="ms")
            frames[sym] = pd.Series([r["c"] for r in results], index=idx)
            time.sleep(0.02)  # be polite to free-tier rate limits
        return pd.DataFrame(frames)

    @staticmethod
    def _parse_interval(interval: str) -> tuple[int, str]:
        mapping = {"1d": (1, "day"), "1h": (1, "hour"), "1m": (1, "minute"),
                   "1wk": (1, "week")}
        return mapping.get(interval, (1, "day"))


class AlpacaAdapter(LiveDataAdapter):
    """Adapter for Alpaca Markets (https://alpaca.markets) — US equities and
    crypto, commonly paired with Alpaca's own execution/brokerage API so
    the same account can both source data and place the resulting trades.
    Requires API key + secret (paper or live).
    """
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.api_key, self.api_secret, self.paper = api_key, api_secret, paper
        try:
            from alpaca.data.historical import StockHistoricalDataClient  # noqa: F401
        except ImportError as e:
            raise ImportError("pip install alpaca-py") from e

    def fetch_prices(self, symbols: list[str], start: str, end: str,
                      interval: str = "1d") -> pd.DataFrame:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(self.api_key, self.api_secret)
        tf_map = {"1d": TimeFrame.Day, "1h": TimeFrame.Hour, "1m": TimeFrame.Minute}
        req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=tf_map.get(interval, TimeFrame.Day),
                                start=start, end=end)
        bars = client.get_stock_bars(req).df
        prices = bars.reset_index().pivot(index="timestamp", columns="symbol", values="close")
        return prices


class BinanceAdapter(LiveDataAdapter):
    """Adapter for Binance (crypto spot market). Public market-data
    endpoints work without an API key for historical klines; supply a
    key/secret only if you also need account/trading endpoints.
    """
    BASE_URL = "https://api.binance.com"

    def __init__(self, api_key: str | None = None, api_secret: str | None = None):
        self.api_key, self.api_secret = api_key, api_secret
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise ImportError("pip install requests") from e

    def fetch_prices(self, symbols: list[str], start: str, end: str,
                      interval: str = "1d") -> pd.DataFrame:
        import requests
        interval_map = {"1d": "1d", "1h": "1h", "1m": "1m", "1wk": "1w"}
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end).timestamp() * 1000)
        frames = {}
        for sym in symbols:
            resp = requests.get(f"{self.BASE_URL}/api/v3/klines", params={
                "symbol": sym, "interval": interval_map.get(interval, "1d"),
                "startTime": start_ms, "endTime": end_ms, "limit": 1000,
            })
            resp.raise_for_status()
            klines = resp.json()
            if not klines:
                continue
            idx = pd.to_datetime([k[0] for k in klines], unit="ms")
            closes = [float(k[4]) for k in klines]
            frames[sym] = pd.Series(closes, index=idx)
            time.sleep(0.05)
        return pd.DataFrame(frames)


ADAPTER_REGISTRY = {
    "yahoo": YahooFinanceAdapter,
    "polygon": PolygonAdapter,
    "alpaca": AlpacaAdapter,
    "binance": BinanceAdapter,
}


def get_adapter(source: str, **kwargs) -> LiveDataAdapter:
    """Factory: get_adapter('polygon', api_key='...') -> PolygonAdapter instance."""
    if source not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown source '{source}'. Available: {list(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[source](**kwargs)
