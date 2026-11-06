import json
from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

import pandas_market_calendars as mcal
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

forecast_tickers = ["NVDA", "GOOGL", "MU", "SPY"]
crypto_names = {"BTC": "Bitcoin", "ETH": "Ethereum", "ZEREBRO": "Zerebro",
                "TRX": "TRON", "HYPE": "Hyperliquid"}
eastern = ZoneInfo("America/New_York")
pacific = ZoneInfo("America/Los_Angeles")
step = pd.Timedelta(minutes=15)

market = mcal.get_calendar("NYSE")

def download_intraday(*, extended=False, crypto_symbol=None):
    symbols = [crypto_symbol] if crypto_symbol else forecast_tickers
    today = pd.Timestamp.now(tz=eastern).date()
    date_range = (
        {"start": today - timedelta(days=7), "end": today + timedelta(days=1)}
        if extended else {"period": "1mo"}
    )
    raw = yf.download(
        symbols, interval="15m", auto_adjust=True, multi_level_index=True,
        prepost=extended, progress=False, ignore_tz=False, **date_range,
    )
    if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
        raise RuntimeError("Yahoo Finance returned no intraday prices. Try refreshing.")
    prices = raw["Close"].reindex(columns=symbols).sort_index()
    prices.index = prices.index.tz_convert(eastern) + step
    return prices.loc[prices.index <= pd.Timestamp.now(tz=eastern)]

def analyze_path(price, now, bars_ahead, *, is_crypto=False):
    horizons = list(range(4, bars_ahead + 1, 4)) if is_crypto and bars_ahead == 96 else [bars_ahead]
    session = price.index.normalize()
    groups = [0] * len(price) if is_crypto else session
    daily = price.groupby(groups)
    times = price.index.to_series().groupby(groups)
    valid = (times.diff() == step) & (times.diff(4) == 4 * step)
    features = pd.DataFrame({
        "Past 15 minutes": daily.pct_change(fill_method=None),
        "Past hour": daily.pct_change(4, fill_method=None),
    }).loc[valid].dropna()

    fits = pd.Series(True, index=price.index)
    if not is_crypto:
        schedule = market.schedule(start_date=price.index[0].date(), end_date=now.date())
        closes = schedule["market_close"].reindex(session.tz_localize(None))
        fits = pd.Series(price.index + bars_ahead * step <= pd.DatetimeIndex(closes), index=price.index)
    targets = pd.DataFrame({
        h: price.reindex(price.index + h * step).to_numpy() / price - 1
        for h in horizons
    }).loc[fits]
    samples = features.join(targets, how="inner").dropna()
    days = samples.index.normalize().unique()
    if len(days) < 5:
        raise RuntimeError("At least five usable trading days are needed for this ticker.")

    split = days[int(len(days) * 0.8)]
    train = samples.loc[samples.index + bars_ahead * step < split]
    test = samples.loc[samples.index >= split]
    if train.empty or test.empty:
        raise RuntimeError("Not enough training and test data for this forecast horizon.")
    columns = features.columns
    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    model.fit(train[columns], train[horizons])
    actual = test[horizons].to_numpy()
    predicted = model.predict(test[columns]).reshape(actual.shape)
    error = 100 * abs(actual[:, -1] - predicted[:, -1]).mean()
    baseline = 100 * abs(actual[:, -1]).mean()

    last = price.index[-1]
    forecast = None
    if pd.Timedelta(0) <= now - last < step and last in features.index and fits.loc[last]:
        model.fit(samples[columns], samples[horizons])
        returns = model.predict(features.loc[[last]]).ravel()
        forecast = pd.Series(
            price.iloc[-1] * (1 + returns),
            index=[last + h * step for h in horizons],
            name="Projected price (USD)",
        )
    return forecast, error, baseline, test.index

st.set_page_config(page_title="Stock Forecast Dashboard", page_icon="📈", layout="wide")
st.title("Stock Forecast Dashboard")
st.caption("15-minute market prices and experimental intraday projections")

@st.cache_data(ttl=240, show_spinner=False)
def load_prices(extended, crypto_symbol=None):
    return download_intraday(extended=extended, crypto_symbol=crypto_symbol)
    
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree as ET

NEWS_QUERIES = {
    "NVDA": ("Nvidia", "Nvidia chip export controls tariffs"),
    "GOOGL": ("Alphabet Google", "Google antitrust regulation"),
    "MU": ("Micron Technology", "Micron chip tariffs export controls"),
    "SPY": ("S&P 500", "US stocks Federal Reserve tariffs government policy"),
    **{ticker: (f"{name} crypto", f"{name} crypto regulation policy")
       for ticker, name in crypto_names.items()},
}

@st.cache_data(ttl=240, show_spinner=False)
def fetch_headlines(query):
    params = urlencode({
        "q": query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    })
    with urlopen(f"https://news.google.com/rss/search?{params}", timeout=8) as response:
        feed = ET.fromstring(response.read())

    headlines = []
    for item in feed.findall("./channel/item"):
        title = item.findtext("title", "").strip()
        url = item.findtext("link", "").strip()
        try:
            published = parsedate_to_datetime(item.findtext("pubDate", ""))
        except (TypeError, ValueError):
            continue
        if title and url.startswith("https://news.google.com/") and published.tzinfo:
            headlines.append((published, title, url))

    return sorted(headlines, key=lambda item: item[0], reverse=True)[:25]


def show_market_news(ticker):
    st.subheader(f"Market news · {ticker}")

    for column, heading, query in zip(
        st.columns(2), ("Company or market", "Politics and policy"), NEWS_QUERIES[ticker]
    ):
        with column:
            st.markdown(f"**{heading}**")
            try:
                headlines = fetch_headlines(query)
            except (OSError, ET.ParseError):
                st.caption("News feed unavailable. Try again on the next refresh.")
                continue
            if not headlines:
                st.caption("No headlines returned.")
            for published, title, url in headlines[:3]:
                local_time = published.astimezone(pacific)
                st.link_button(title, url)
                st.caption(f"{local_time:%b %d at %I:%M %p %Z}")
                
def add_news_markers(fig, ticker, prices, currency):
    colors = {"Company or market": "#22c55e", "Politics and policy": "#d946ef"}
    marker_count = 0
    seen = set()
    for (category, color), query in zip(colors.items(), NEWS_QUERIES[ticker]):
        try:
            headlines = fetch_headlines(query)
        except (OSError, ET.ParseError):
            continue
        times, levels, titles = [], [], []
        for published, title, url in headlines:
            published = pd.Timestamp(published)
            if url in seen or not prices.index[0] <= published <= prices.index[-1]:
                continue
            seen.add(url)
            times.append(published.tz_convert(pacific).tz_localize(None))
            bar = prices.index.searchsorted(published, side="right") - 1
            levels.append(prices.iloc[bar])
            titles.append(title)
            if len(times) == 5:
                break
        if times:
            fig.add_scatter(
                x=times, y=levels, text=titles, name=category, mode="markers",
                marker={"symbol": "diamond", "size": 11, "color": color},
                hovertemplate="%{text}<br>Previous observed bar: %{y:,.6g} " + currency + "<extra>%{fullData.name}</extra>",
            )
            marker_count += len(times)
    return marker_count


def show_streaming_chart(ticker, extended, *, overnight=False):
    st.subheader(f"{'Overnight market' if overnight else 'Streaming chart'} · {ticker}")
    st.caption("One-minute bars from TradingView. Availability and delay depend on the feed.")
    exchange = "BOATS" if overnight else ("AMEX" if ticker == "SPY" else "NASDAQ")
    symbol = f"{exchange}:{ticker}"
    settings = json.dumps({
        "autosize": True, "symbol": symbol, "interval": "1",
        "timezone": "America/Los_Angeles", "theme": "dark", "style": "1",
        "locale": "en", "allow_symbol_change": False,
        "support_host": "https://www.tradingview.com",
        "extended_hours": extended,
    })
    components.html(f'''
        <div class="tradingview-widget-container" style="height:510px;width:100%">
          <div class="tradingview-widget-container__widget" style="height:480px;width:100%"></div>
          <div class="tradingview-widget-copyright">
            <a href="https://www.tradingview.com/symbols/{symbol.replace(':', '-')}/" target="_blank" rel="noopener noreferrer">{ticker} chart on TradingView</a>
          </div>
          <script src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>{settings}</script>
        </div>
    ''', height=520)


@st.cache_data(ttl=55, show_spinner=False)
def load_minute_prices(ticker, extended):
    raw = yf.download(ticker, period="5d", interval="1m", auto_adjust=True,
                      prepost=extended, progress=False, ignore_tz=False,
                      multi_level_index=False, timeout=8)
    if raw is None or raw.empty or "Close" not in raw:
        return pd.Series(dtype=float)
    prices = raw["Close"].dropna().sort_index()
    prices.index = prices.index.tz_convert(eastern) + pd.Timedelta(minutes=1)
    return prices.loc[prices.index <= pd.Timestamp.now(tz=eastern)]


@st.cache_data(ttl=55, show_spinner=False)
def load_kraken_prices(ticker, interval):
    params = urlencode({"pair": f"{ticker}USD", "interval": interval})
    with urlopen(f"https://api.kraken.com/0/public/OHLC?{params}", timeout=8) as response:
        payload = json.load(response)
    if payload.get("error"):
        raise RuntimeError(f"Kraken returned no usable bars for {ticker}/USD.")
    rows = next((value for key, value in payload.get("result", {}).items() if key != "last"), [])
    if not rows:
        return pd.Series(dtype=float)
    ends = pd.to_datetime([row[0] for row in rows], unit="s", utc=True).tz_convert(eastern)
    ends += pd.Timedelta(minutes=interval)
    prices = pd.Series([float(row[4]) for row in rows], index=ends).sort_index()
    return prices.loc[prices.index <= pd.Timestamp.now(tz=eastern)]


@st.cache_data(ttl=900, show_spinner=False)
def load_idr_rate():
    quotes = yf.download("IDR=X", period="5d", interval="1d", progress=False,
                         auto_adjust=False, multi_level_index=False, timeout=8)
    if quotes is None or quotes.empty or "Close" not in quotes:
        raise RuntimeError("USD/IDR rate unavailable.")
    closes = quotes["Close"].dropna().sort_index()
    if closes.empty or not 0 < closes.iloc[-1] < float("inf"):
        raise RuntimeError("USD/IDR rate unavailable.")
    return float(closes.iloc[-1]), closes.index[-1].date()


def show_market_summary(recent, currency, path, regular, hours, error, baseline):
    st.subheader("Market summary")
    window = recent.loc[recent.index[-1] - pd.Timedelta(hours=4):]
    if len(window) >= 4:
        prefix = "Rp " if currency == "IDR" else "USD "
        low, high = window.min(), window.max()
        precision = 2 if low >= 10 else 6
        levels = st.columns(2)
        levels[0].metric("Support area", f"{prefix}{low:,.{precision}f}")
        levels[1].metric("Resistance area", f"{prefix}{high:,.{precision}f}")
        st.caption("Support is an area where a decline may slow. Resistance is an area where a rally may stall. These estimates use the lowest and highest observed bar closes over up to four hours on this chart.")
    else:
        st.caption("Too few recent prices to estimate support and resistance.")

    if path is None:
        st.markdown("**Wait · No fresh forecast**")
        return

    change = 100 * (path.iloc[-1] / regular.iloc[-1] - 1)
    if error >= baseline:
        signal = "Hold / wait"
        reason = "The model did not beat the no-change guess in its historical test."
    elif abs(change) <= error:
        signal = "Hold / wait"
        reason = "The estimated move is within the model’s average historical miss."
    elif change > 0:
        signal = "Potential buy setup"
        reason = "The model beat the no-change guess, and the projected gain exceeds its average historical miss."
    else:
        signal = "Potential sell setup"
        reason = "The model beat the no-change guess, and the projected decline exceeds its average historical miss."
    st.markdown(f"**{signal}** · {hours}-hour forecast: **{change:+.2f}%** from the model’s starting price.")
    st.caption(f"{reason} This is a model-based signal. It does not account for fees, spreads, or your positions.")


def track_forecasts(ticker, hours, regular, path, now):
    history = st.session_state.setdefault("forecast_history", {})
    key = (ticker, hours)
    entries = history.setdefault(key, {})
    anchor = regular.index[-1]
    if path is not None and path.index[-1] > now and anchor not in entries:
        entries[anchor] = {"start": float(regular.iloc[-1]), "path": path.copy(), "actual": None}
        if len(entries) > 192:
            del entries[min(entries)]
    for entry in entries.values():
        target = entry["path"].index[-1]
        if entry["actual"] is None and target <= now and target in regular.index:
            entry["actual"] = float(regular.loc[target])
    return entries


def show_forecast_history(entries, now, currency, rate):
    st.subheader("Forecast history")
    st.caption("Saved from forecasts shown in this browser session. The actual price comes from the completed 15-minute bar at the target time. Times are Los Angeles time.")
    if not entries:
        st.caption("No forecasts have been captured yet for this ticker and horizon.")
        return
    rows = []
    newest = sorted(entries, reverse=True)
    pending = [a for a in newest if entries[a]["path"].index[-1] > now]
    expired = [a for a in newest if entries[a]["path"].index[-1] <= now]
    for anchor in expired[:12] + pending[:6]:
        entry = entries[anchor]
        target = entry["path"].index[-1]
        forecast = float(entry["path"].iloc[-1])
        actual = entry["actual"]
        status = "Compared" if actual is not None else ("Awaiting price bar" if target <= now else "Pending")
        precision = 2 if forecast * rate >= 1 else 6
        rows.append({
            "Started": anchor.tz_convert(pacific).strftime("%b %d %I:%M %p %Z"),
            "Target time": target.tz_convert(pacific).strftime("%b %d %I:%M %p %Z"),
            f"Predicted ({currency})": f"{forecast * rate:,.{precision}f}",
            f"Actual ({currency})": f"{actual * rate:,.{precision}f}" if actual is not None else "—",
            "Absolute miss": f"{100 * abs(actual - forecast) / entry['start']:.3f} pp" if actual is not None else "—",
            "Status": status,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


@st.fragment(run_every="1m")
def dashboard():
    controls = st.columns([2, 2, 2, 1])
    ticker = controls[0].selectbox("Ticker", [*forecast_tickers, *crypto_names])
    is_crypto = ticker in crypto_names
    quote_symbol = f"{ticker}-USD" if is_crypto else ticker
    if ticker in {"HYPE", "ZEREBRO"}:
        controls[0].link_button(f"{ticker} on Kraken", f"https://www.kraken.com/prices/{crypto_names[ticker].lower()}")
    else:
        controls[0].link_button(f"{ticker} on Yahoo Finance", f"https://finance.yahoo.com/quote/{quote_symbol}/")
    horizons = [1, 2, 24] if is_crypto else [1, 2]
    hours = controls[1].selectbox("Forecast horizon", horizons, format_func=lambda h: f"{h} hour" if h == 1 else f"{h} hours")
    currency = controls[2].selectbox("Display currency", ["USD", "IDR"])
    if is_crypto:
        controls[2].caption("Crypto · 24/7")
    extended = not is_crypto and controls[2].checkbox("Include extended hours", value=True)
    if controls[3].button("Refresh prices"):
        load_prices.clear()
        fetch_headlines.clear()
        load_minute_prices.clear()
        load_kraken_prices.clear()
        load_idr_rate.clear()
    if not is_crypto:
        with st.expander(f"Overnight market · {ticker}"):
            st.caption("Blue Ocean ATS overnight prices. The amber model uses regular-session data.")
            show_streaming_chart(ticker, extended=False, overnight=True)
    st.subheader("Latest prices and forecast")
    source = "Yahoo Finance"
    try:
        with st.spinner("Checking for the latest completed bars..."):
            if ticker in {"HYPE", "ZEREBRO"}:
                source = "Kraken"
                regular = load_kraken_prices(ticker, 15)
            else:
                regular = load_prices(False, quote_symbol if is_crypto else None)[quote_symbol].dropna()
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(f"{source} price refresh failed. {exc}")
        return
    if regular.empty:
        st.warning(f"No 15-minute prices returned for {ticker}.")
        return

    observed = regular
    if extended:
        try:
            extra = load_prices(True)[ticker].dropna()
            observed = regular.combine_first(extra).sort_index()
        except (RuntimeError, ValueError, OSError):
            st.warning("Extended-hours prices are temporarily unavailable.")
    bar_size = "15-minute"
    try:
        minute_prices = (load_kraken_prices(ticker, 1) if source == "Kraken"
                         else load_minute_prices(quote_symbol, extended))
        if not minute_prices.empty and minute_prices.index[-1] >= observed.index[-1]:
            observed = minute_prices
            bar_size = "1-minute"
    except (RuntimeError, ValueError, OSError):
        pass
    now = pd.Timestamp.now(tz=eastern)
    latest = observed.index[-1]
    anchor = regular.index[-1]
    if latest.date() != now.date():
        st.info(f"No completed bars for today yet. Showing {latest:%B %d, %Y}.")
    st.caption(f"Latest chart bar {latest.tz_convert(pacific):%b %d, %Y at %I:%M %p %Z} · {source} {bar_size} bars · Checks every minute while open")

    path = None
    error = baseline = None
    try:
        path, error, baseline, test_dates = analyze_path(regular, now, hours * 4, is_crypto=is_crypto)
    except RuntimeError as exc:
        st.info(str(exc))
    else:
        columns = st.columns(3)
        columns[0].metric("Model error", f"{error:.3f} pp")
        columns[1].metric("Flat-price baseline error", f"{baseline:.3f} pp")
        columns[2].metric("Improvement", f"{baseline - error:+.3f} pp")
        st.caption(f"Historical test {test_dates[0]:%b %d} to {test_dates[-1]:%b %d} · {len(test_dates)} examples · Error measured at the selected horizon")

    forecast_history = track_forecasts(ticker, hours, regular, path, now)
    rate = 1.0
    if currency == "IDR":
        try:
            rate, rate_date = load_idr_rate()
        except (RuntimeError, ValueError, OSError):
            st.warning("USD/IDR rate unavailable. Showing USD prices.")
            currency = "USD"
        else:
            st.caption(f"Conversion: 1 USD ≈ Rp {rate:,.2f} ({rate_date:%b %d}, Yahoo Finance). The same rate is applied to the entire chart.")

    if path is not None:
        target_time = path.index[-1].tz_convert(pacific)
        target_price = path.iloc[-1]
        change = 100 * (target_price / regular.iloc[-1] - 1)
        displayed_target = target_price * rate
        price_format = ",.2f" if displayed_target >= 1 else ",.6f"
        prefix = "Rp " if currency == "IDR" else "$"
        st.metric(f"{hours}-hour price estimate", f"{prefix}{displayed_target:{price_format}}",
                  f"{change:+.2f}% from the forecast start")
        st.caption(f"Target time {target_time:%b %d at %I:%M %p %Z} · {hours} hour(s) after the last completed model bar")

    window_start = latest - pd.Timedelta(hours=24) if is_crypto else latest.normalize()
    recent = observed.loc[window_start:] * rate
    show_market_summary(recent, currency, path, regular, hours, error, baseline)
    fig = go.Figure(go.Scatter(
        x=recent.index.tz_convert(pacific).tz_localize(None), y=recent, name=f"Observed ({bar_size})",
        mode="lines+markers", line={"color": "#22d3ee", "width": 2}, marker={"size": 4},
    ))
    if path is not None:
        projected_times = pd.DatetimeIndex([anchor, *path.index]).tz_convert(pacific).tz_localize(None)
        fig.add_scatter(
            x=projected_times, y=[regular.iloc[-1] * rate, *(path * rate)], name=f"{hours}-hour forecast",
            mode="lines+markers", line={"color": "#f59e0b", "width": 3, "dash": "dash"},
        )
    settled = [a for a, entry in forecast_history.items()
               if entry["path"].index[-1] >= recent.index[0] and entry["actual"] is not None]
    if settled:
        saved_anchor = max(settled)
        saved = forecast_history[saved_anchor]
        saved_path = saved["path"]
        saved_times = pd.DatetimeIndex([saved_anchor, *saved_path.index]).tz_convert(pacific).tz_localize(None)
        fig.add_scatter(
            x=saved_times, y=[saved["start"] * rate, *(saved_path * rate)],
            name="Last completed forecast", mode="lines+markers",
            line={"color": "#f59e0b", "width": 2, "dash": "dot"}, opacity=0.8,
        )
    fig.update_layout(
        title=f"{ticker} · {latest:%B %d, %Y}", yaxis_title=f"Price ({currency})",
        xaxis_title="Los Angeles time (Pacific)", hovermode="x unified", height=455,
        legend={"orientation": "h", "y": 1.10},
    )
    fig.update_xaxes(tickformat="%I:%M %p")
    marker_count = add_news_markers(fig, ticker, recent, currency)
    st.plotly_chart(fig, width="stretch")
    st.markdown(":blue[**Solid cyan**] · Observed prices &nbsp;&nbsp; :orange[**Dashed amber**] · Forecast")
    if marker_count:
        st.caption("Hover over a diamond to see the headline at its publication time. The marker uses the last observed price at that time. Timing alone cannot establish what caused a move.")
    else:
        st.caption("No matching headlines were found within the displayed price window. See the latest headlines below.")
    show_forecast_history(forecast_history, now, currency, rate)
    show_market_news(ticker)
    if path is not None:
        st.caption(f"Forecast starts at the completed 15-minute bar at {anchor.tz_convert(pacific):%I:%M %p %Z}." + (" The dashed path shows one separate estimate per hour." if hours == 24 else f" The dashed segment connects the starting price to a single {hours}-hour estimate."))
    if path is None:
        if is_crypto:
            st.caption("The forecast needs a fresh completed 15-minute bar and one hour of continuous recent history.")
        else:
            st.caption(f"A fresh forecast appears when the model has one hour of regular-session history and the full {hours * 60}-minute projection fits before the close.")
    if not is_crypto:
        with st.expander("TradingView streaming chart"):
            show_streaming_chart(ticker, extended)


dashboard()
