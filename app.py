from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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

@st.cache_data(ttl=840, show_spinner=False)
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

    return sorted(headlines, key=lambda item: item[0], reverse=True)[:3]


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
            for published, title, url in headlines:
                local_time = published.astimezone(pacific)
                st.link_button(title, url)
                st.caption(f"{local_time:%b %d at %I:%M %p %Z}")
                
@st.fragment(run_every="15m")
def dashboard():
    controls = st.columns([2, 2, 2, 1])
    ticker = controls[0].selectbox("Ticker", forecast_tickers)
    hours = controls[1].selectbox("Forecast horizon", [1, 2], format_func=lambda h: f"{h} hour" if h == 1 else f"{h} hours")
    extended = controls[2].checkbox("Include extended hours", value=True)
    if controls[3].button("Refresh prices"):
        load_prices.clear()
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
    now = pd.Timestamp.now(tz=eastern)
    latest = observed.index[-1]
    anchor = regular.index[-1]
    if latest.date() != now.date():
        st.info(f"No completed bars for today yet. Showing {latest:%B %d, %Y}.")
    st.caption(f"Latest chart bar {latest.tz_convert(pacific):%b %d, %Y at %I:%M %p %Z} · Yahoo Finance · Refreshes every 15 minutes while open")

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
        x=recent.index.tz_convert(pacific).tz_localize(None), y=recent, name="Observed",
        mode="lines+markers", line={"color": "#2563eb"}, marker={"size": 4},
    ))
    if path is not None:
        projected_times = pd.DatetimeIndex([anchor, *path.index]).tz_convert(pacific).tz_localize(None)
        fig.add_scatter(
            x=projected_times, y=[regular.iloc[-1], *path], name="Projected",
            mode="lines+markers", line={"color": "#b45309", "dash": "dash"},
        )
    fig.update_layout(
        title=f"{ticker} · {latest:%B %d, %Y}", yaxis_title="Price (USD)",
        xaxis_title="Los Angeles time (Pacific)", hovermode="x unified", height=455,
        legend={"orientation": "h", "y": 1.10},
    )
    fig.update_xaxes(tickformat="%I:%M %p")
    st.plotly_chart(fig, width="stretch")
    show_market_news(ticker)
    if path is None:
        st.caption(f"A fresh forecast appears when the model has one hour of regular-session history and the full {hours * 60}-minute projection fits before the close.")

dashboard()
