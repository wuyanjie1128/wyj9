# streamlit_app.py
# KOSPI 200 Stock Recommendation System (English UI)
# Data provider: Twelve Data (https://twelvedata.com) – requires API key.
# Put your API key in Streamlit Secrets (see instructions below).
#
# DISCLAIMER: This is an educational demo, not financial advice.

import os
import time
import json
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# -------------------------------
# Configuration & Secrets
# -------------------------------
# Preferred: set in .streamlit/secrets.toml as:
# [twelvedata]
# api_key = "YOUR_API_KEY"
TD_API_KEY = st.secrets.get("twelvedata", {}).get("api_key") or os.getenv("TWELVEDATA_API_KEY", "")

TD_BASE = "https://api.twelvedata.com"

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="KOSPI 200 Stock Recommendation System", layout="wide")

st.title("KOSPI 200 Stock Recommendation System")
st.caption("Built with Streamlit + Twelve Data API • Educational demo only (not investment advice)")

with st.sidebar:
    st.header("🔐 API Authentication")
    if TD_API_KEY:
        st.success("Twelve Data API key is set.")
    else:
        st.error("No Twelve Data API key found. Add it to Streamlit Secrets or as env var TWELVEDATA_API_KEY.")
        st.info("Sidebar → \"⋮\" → Edit secrets → add under [twelvedata] api_key = \"...\"")

    st.header("⚙️ Parameters")
    lookback_days = st.slider("Indicators lookback (days)", min_value=60, max_value=400, value=200, step=20)
    max_symbols = st.slider("Max symbols to analyze (rate-limit safety)", min_value=10, max_value=200, value=60, step=10)
    top_k = st.slider("Top N recommendations", min_value=3, max_value=20, value=5, step=1)
    st.divider()
    st.markdown("**Scoring (weights):**")
    w_price_vs_sma50 = st.number_input("Price above SMA50", value=2.0, step=0.5)
    w_trend_sma20_gt_sma50 = st.number_input("SMA20 > SMA50", value=1.0, step=0.5)
    w_mom_20d = st.number_input("20d momentum (scaled)", value=1.0, step=0.5)
    w_rsi_band = st.number_input("RSI(14) ideal band (50–70)", value=1.5, step=0.5)
    w_vol_penalty = st.number_input("Volatility penalty", value=1.0, step=0.5)

st.info(
    "Tip: For a first run, keep `Max symbols` modest to avoid hitting free-tier rate limits. "
    "You can increase gradually."
)

# -------------------------------
# Utilities
# -------------------------------
@st.cache_data(show_spinner=False)
def fetch_kospi200_symbols_from_wikipedia() -> pd.DataFrame:
    """
    Fetch KOSPI 200 components from Wikipedia.
    We only need the 6-digit numeric codes for XKRX, which match many data providers.
    If parsing changes, you can replace this with a local CSV.
    """
    url = "https://en.wikipedia.org/wiki/KOSPI_200"
    tables = pd.read_html(url)
    # Heuristic: the table with 'Company' and 'Symbol' columns
    candidates = [t for t in tables if {"Company", "Symbol"}.issubset(set(t.columns))]
    if not candidates:
        raise RuntimeError("Could not find KOSPI 200 table on Wikipedia.")
    df = candidates[0].copy()
    # Normalize symbol to 6-digit strings where possible
    df["Symbol"] = df["Symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    df = df.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
    df = df.rename(columns={"Company": "name", "Symbol": "symbol"})
    return df[["symbol", "name"]]

@st.cache_data(show_spinner=False)
def fallback_symbols() -> pd.DataFrame:
    """
    Minimal fallback list in case Wikipedia layout changes or network issues occur.
    These are well-known KOSPI names (not the full 200).
    """
    data = [
        {"symbol": "005930", "name": "Samsung Electronics"},
        {"symbol": "000660", "name": "SK hynix"},
        {"symbol": "035420", "name": "NAVER"},
        {"symbol": "035720", "name": "Kakao"},
        {"symbol": "051910", "name": "LG Chem"},
        {"symbol": "005380", "name": "Hyundai Motor"},
        {"symbol": "207940", "name": "Samsung Biologics"},
        {"symbol": "068270", "name": "Celltrion"},
        {"symbol": "105560", "name": "KB Financial Group"},
        {"symbol": "096770", "name": "SK Innovation"},
    ]
    return pd.DataFrame(data)

def td_headers():
    # Twelve Data supports apikey in query string and/or in headers.
    # We'll also send it via header for clarity.
    return {"Authorization": f"apikey {TD_API_KEY}"} if TD_API_KEY else {}

def td_get(path: str, params: dict) -> dict:
    """HTTP GET to Twelve Data API with simple error handling."""
    params = dict(params or {})
    if TD_API_KEY and "apikey" not in params:
        params["apikey"] = TD_API_KEY
    url = f"{TD_BASE.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, params=params, headers=td_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()
    # Twelve Data error format often under "status":"error" or "code"/"message"
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {data.get('message')}")
    return data

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(show_spinner=False)
def fetch_daily_timeseries(symbol: str, lookback_days: int) -> pd.DataFrame:
    """
    Pull daily OHLCV via Twelve Data time_series endpoint for a given XKRX symbol (6-digit).
    """
    # We ask for more bars than lookback to safely compute indicators.
    out_size = max(lookback_days + 50, 150)
    params = {
        "symbol": symbol,         # e.g., "005930" for Samsung Electronics
        "interval": "1day",
        "outputsize": out_size,
        "format": "JSON",
    }
    data = td_get("/time_series", params)
    if "values" not in data:
        raise RuntimeError(f"No time_series values for {symbol}: {data}")
    df = pd.DataFrame(data["values"])
    # Normalize dtypes
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def score_symbol(ts: pd.DataFrame) -> dict:
    """
    Compute indicators & a transparent score from the time series.
    Returns dict with latest metrics and a 'score' float.
    """
    close = ts["close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi14 = compute_rsi(close, 14)

    if len(ts) < 60 or sma50.isna().all():
        return None

    latest = ts.iloc[-1]
    latest_close = float(latest["close"])
    latest_date = latest["datetime"]

    last20 = close.iloc[-21] if len(close) > 21 else np.nan
    mom_20d = (latest_close / last20 - 1.0) if pd.notna(last20) and last20 != 0 else np.nan

    # simple volatility proxy over 20d
    vol20 = close.pct_change().rolling(20).std().iloc[-1]

    _sma20 = float(sma20.iloc[-1])
    _sma50 = float(sma50.iloc[-1])
    _rsi14 = float(rsi14.iloc[-1])

    # Scoring (weights configurable in sidebar)
    s = 0.0
    s += w_price_vs_sma50 if latest_close > _sma50 else 0.0
    s += w_trend_sma20_gt_sma50 if _sma20 > _sma50 else 0.0

    if not np.isnan(mom_20d):
        # Scale momentum by 100 bp units for interpretability (e.g., 0.05 -> 5)
        s += w_mom_20d * (mom_20d * 100)

    # Reward RSI in [50, 70]
    if 50 <= _rsi14 <= 70:
        s += w_rsi_band
    elif _rsi14 > 75:
        s -= 0.5  # overbought small penalty
    elif _rsi14 < 30:
        s -= 0.5  # oversold small penalty

    if not np.isnan(vol20):
        # Penalize excessive short-term volatility
        s -= w_vol_penalty * (vol20 * 100)

    return {
        "date": latest_date.date().isoformat(),
        "close": latest_close,
        "sma20": _sma20,
        "sma50": _sma50,
        "rsi14": _rsi14,
        "mom_20d_pct": mom_20d * 100 if not np.isnan(mom_20d) else np.nan,
        "vol20_pct": vol20 * 100 if not np.isnan(vol20) else np.nan,
        "score": s,
    }

# -------------------------------
# Main flow
# -------------------------------
st.subheader("Universe: KOSPI 200 components (auto-fetched)")
try:
    kospi_df = fetch_kospi200_symbols_from_wikipedia()
    st.success(f"Fetched {len(kospi_df)} symbols from Wikipedia.")
except Exception as e:
    st.warning(f"Auto-fetch failed ({e}). Using fallback sample list.")
    kospi_df = fallback_symbols()

# Rate-limit safety cap
universe = kospi_df.head(max_symbols).copy()

st.write(
    "Symbols under analysis (first rows shown):",
)
st.dataframe(universe.head(20), use_container_width=True)

if not TD_API_KEY:
    st.stop()

st.subheader("Run analysis")
run_btn = st.button("Analyze & Rank")
if run_btn:
    rows = []
    progress = st.progress(0)
    for i, row in universe.iterrows():
        sym = row["symbol"]
        name = row["name"]
        try:
            ts = fetch_daily_timeseries(sym, lookback_days=lookback_days)
            metrics = score_symbol(ts)
            if metrics is None:
                continue
            rows.append({
                "symbol": sym,
                "name": name,
                **metrics
            })
        except Exception as e:
            # In free tiers you may hit rate limits; brief backoff helps.
            time.sleep(0.3)
        finally:
            progress.progress(int((i + 1) / len(universe) * 100))

    if not rows:
        st.error("No results computed. Try lowering 'Max symbols' or increasing lookback.")
        st.stop()

    results = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    st.success(f"Done. Computed scores for {len(results)} symbols.")

    # Top N table
    st.subheader(f"Top {top_k} Recommendations")
    st.dataframe(results.head(top_k), use_container_width=True)

    # Detail viewer
    st.subheader("Details & Chart")
    pick = st.selectbox(
        "Select a symbol to visualize",
        results["symbol"].head(top_k).tolist()
    )

    if pick:
        detail_row = results[results["symbol"] == pick].iloc[0]
        st.markdown(
            f"**{detail_row['symbol']} — {detail_row['name']}**  "
            f"(Last date: {detail_row['date']})"
        )
        st.write({
            "close": round(detail_row["close"], 2),
            "SMA20": round(detail_row["sma20"], 2),
            "SMA50": round(detail_row["sma50"], 2),
            "RSI(14)": round(detail_row["rsi14"], 2),
            "20d momentum %": round(detail_row["mom_20d_pct"], 2) if pd.notna(detail_row["mom_20d_pct"]) else None,
            "20d vol % (stdev)": round(detail_row["vol20_pct"], 2) if pd.notna(detail_row["vol20_pct"]) else None,
            "score": round(detail_row["score"], 2),
        })

        # Chart
        try:
            ts = fetch_daily_timeseries(pick, lookback_days=lookback_days)
            ts = ts.tail(lookback_days).copy()
            ts["SMA20"] = ts["close"].rolling(20).mean()
            ts["SMA50"] = ts["close"].rolling(50).mean()

            import altair as alt
            base = alt.Chart(ts).encode(x="datetime:T")

            price_line = base.mark_line().encode(y=alt.Y("close:Q", title="Price"))
            sma20_line = base.mark_line(strokeDash=[4,2]).encode(y="SMA20:Q", color=alt.value("#999"))
            sma50_line = base.mark_line(strokeDash=[2,2]).encode(y="SMA50:Q", color=alt.value("#555"))

            st.altair_chart((price_line + sma20_line + sma50_line).properties(height=320), use_container_width=True)
        except Exception as e:
            st.warning(f"Charting failed: {e}")

    st.caption("Scoring is heuristic. Adjust weights in the sidebar to explore different regimes.")
