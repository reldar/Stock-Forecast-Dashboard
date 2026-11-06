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
eastern = ZoneInfo("America/New_York")
pacific = ZoneInfo("America/Los_Angeles")
step = pd.Timedelta(minutes=15)

market = mcal.get_calendar("NYSE")

def download_intraday(*, extended=False):
    today = pd.Timestamp.now(tz=eastern).date()
    date_range = (
        {"start": today - timedelta(days=7), "end": today + timedelta(days=1)}
        if extended else {"period": "1mo"}
    )
    raw = yf.download(
        forecast_tickers, interval="15m", auto_adjust=True,
        prepost=extended, progress=False, ignore_tz=False, **date_range,
    )
    if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
        raise RuntimeError("Yahoo Finance returned no intraday prices. Try refreshing.")
    prices = raw["Close"].reindex(columns=forecast_tickers).sort_index()
    prices.index = prices.index.tz_convert(eastern) + step
    return prices.loc[prices.index <= pd.Timestamp.now(tz=eastern)]

def analyze_path(price, now, bars_ahead):
    horizons = list(range(1, bars_ahead + 1))
    session = price.index.normalize()
    daily = price.groupby(session)
    times = price.index.to_series().groupby(session)
    valid = (times.diff() == step) & (times.diff(4) == 4 * step)
    features = pd.DataFrame({
        "Past 15 minutes": daily.pct_change(fill_method=None),
        "Past hour": daily.pct_change(4, fill_method=None),
    }).loc[valid].dropna()

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
    train = samples.loc[samples.index < split]
    test = samples.loc[samples.index >= split]
    columns = features.columns
    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    model.fit(train[columns], train[horizons])
    actual = test[horizons].to_numpy()
    error = 100 * abs(actual - model.predict(test[columns])).mean()
    baseline = 100 * abs(actual).mean()

    last = price.index[-1]
    forecast = None
    if pd.Timedelta(0) <= now - last < step and last in features.index and fits.loc[last]:
        model.fit(samples[columns], samples[horizons])
        returns = model.predict(features.loc[[last]])[0]
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
def load_prices(extended):
    return download_intraday(extended=extended)
    
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree as ET

NEWS_QUERIES = {
    "NVDA": ("Nvidia", "Nvidia chip export controls tariffs"),
    "GOOGL": ("Alphabet Google", "Google antitrust regulation"),
    "MU": ("Micron Technology", "Micron chip tariffs export controls"),
    "SPY": ("S&P 500", "US stocks Federal Reserve tariffs government policy"),
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
                
def add_news_markers(fig, ticker, prices):
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
                hovertemplate="%{text}<br>Previous observed bar: $%{y:.2f}<extra>%{fullData.name}</extra>",
            )
            marker_count += len(times)
    return marker_count


def show_streaming_chart(ticker, extended):
    st.subheader(f"Streaming chart · {ticker}")
    st.caption("One-minute bars from TradingView. Availability and delay depend on the feed.")
    symbol = ("AMEX:" if ticker == "SPY" else "NASDAQ:") + ticker
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


@st.fragment(run_every="1m")
def dashboard():
    controls = st.columns([2, 2, 2, 1])
    ticker = controls[0].selectbox("Ticker", forecast_tickers)
    controls[0].link_button(f"{ticker} on Yahoo Finance", f"https://finance.yahoo.com/quote/{ticker}/")
    hours = controls[1].selectbox("Forecast horizon", [1, 2], format_func=lambda h: f"{h} hour" if h == 1 else f"{h} hours")
    extended = controls[2].checkbox("Include extended hours", value=True)
    if controls[3].button("Refresh prices"):
        load_prices.clear()
        fetch_headlines.clear()
        load_minute_prices.clear()
    st.subheader("Latest prices and forecast")
    try:
        with st.spinner("Checking for the latest completed bars..."):
            regular = load_prices(False)[ticker].dropna()
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(f"Price refresh failed. {exc}")
        return
    if regular.empty:
        st.warning(f"No regular-session prices returned for {ticker}.")
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
        minute_prices = load_minute_prices(ticker, extended)
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
    st.caption(f"Latest chart bar {latest.tz_convert(pacific):%b %d, %Y at %I:%M %p %Z} · Yahoo Finance {bar_size} bars · Checks every minute while open")

    path = None
    try:
        path, error, baseline, test_dates = analyze_path(regular, now, hours * 4)
    except RuntimeError as exc:
        st.info(str(exc))
    else:
        columns = st.columns(3)
        columns[0].metric("Model error", f"{error:.3f} pp")
        columns[1].metric("Flat-price baseline error", f"{baseline:.3f} pp")
        columns[2].metric("Improvement", f"{baseline - error:+.3f} pp")
        st.caption(f"Historical test {test_dates[0]:%b %d} to {test_dates[-1]:%b %d} · {len(test_dates)} examples · Lower error is better")

    recent = observed.loc[latest.normalize():]
    fig = go.Figure(go.Scatter(
        x=recent.index.tz_convert(pacific).tz_localize(None), y=recent, name=f"Observed ({bar_size})",
        mode="lines+markers", line={"color": "#22d3ee", "width": 2}, marker={"size": 4},
    ))
    if path is not None:
        projected_times = pd.DatetimeIndex([anchor, *path.index]).tz_convert(pacific).tz_localize(None)
        fig.add_scatter(
            x=projected_times, y=[regular.iloc[-1], *path], name=f"{hours}-hour forecast",
            mode="lines+markers", line={"color": "#f59e0b", "width": 3, "dash": "dash"},
        )
    fig.update_layout(
        title=f"{ticker} · {latest:%B %d, %Y}", yaxis_title="Price (USD)",
        xaxis_title="Los Angeles time (Pacific)", hovermode="x unified", height=455,
        legend={"orientation": "h", "y": 1.10},
    )
    fig.update_xaxes(tickformat="%I:%M %p")
    marker_count = add_news_markers(fig, ticker, recent)
    st.plotly_chart(fig, width="stretch")
    st.markdown(":blue[**Solid cyan**] · Observed prices &nbsp;&nbsp; :orange[**Dashed amber**] · Forecast")
    if marker_count:
        st.caption("Hover over a diamond to see the headline at its publication time. The marker uses the last observed price at that time. Timing alone cannot establish what caused a move.")
    else:
        st.caption("No matching headlines were found within the displayed price window. See the latest headlines below.")
    show_market_news(ticker)
    if path is not None:
        st.caption(f"Forecast starts at the completed 15-minute bar at {anchor.tz_convert(pacific):%I:%M %p %Z}. Newer observed prices may overlap the first part of the forecast.")
    if path is None:
        st.caption(f"A fresh forecast appears when the model has one hour of regular-session history and the full {hours * 60}-minute projection fits before the close.")
    with st.expander("TradingView streaming chart"):
        show_streaming_chart(ticker, extended)


dashboard()
