# streamlit_app.py
# KOSPI 200 Stock Recommendation System (English UI)
# Left sidebar shows Korean-labeled settings you requested.
# Data provider: Twelve Data API (API-key auth).
# DISCLAIMER: Educational demo only, not financial advice.

import os
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# -------------------------------
# Secrets / Config
# -------------------------------
# Twelve Data API key from secrets or env
TD_API_KEY = st.secrets.get("twelvedata", {}).get("api_key") or os.getenv("TWELVEDATA_API_KEY", "")
TD_BASE = "https://api.twelvedata.com"

# Optional broker-style credentials from secrets (for your sidebar defaults)
BROKER_APP_KEY = st.secrets.get("broker", {}).get("app_key", "")
BROKER_APP_SECRET = st.secrets.get("broker", {}).get("app_secret", "")
BROKER_ACCOUNT = st.secrets.get("broker", {}).get("account", "")

st.set_page_config(page_title="KOSPI 200 Stock Recommendation System", layout="wide")
st.title("KOSPI 200 Stock Recommendation System")
st.caption("Streamlit + Twelve Data API • Educational demo (not investment advice)")

# -------------------------------
# Sidebar (Korean labels)
# -------------------------------
with st.sidebar:
    st.header("설정")

    st.subheader("🔑 API 인증 정보")
    app_key = st.text_input("APP KEY", value=BROKER_APP_KEY)
    app_secret = st.text_input("APP SECRET", value=BROKER_APP_SECRET, type="password")
    account_no = st.text_input("계좌번호", value=BROKER_ACCOUNT)

    # Twelve Data key status (for market data)
    if TD_API_KEY:
        st.success("Twelve Data API key detected.")
    else:
        st.error("No Twelve Data API key. Set in secrets or env: TWELVEDATA_API_KEY")

    st.subheader("📊 분석 설정")
    # number of recommendations (3–10)
    top_k = st.slider("추천받을 종목 개수", min_value=3, max_value=10, value=5, step=1)
    # minimum trading size (KRW 100M units = 억원)
    min_trading_ogwon = st.number_input("최소 거래 규모 (억원)", min_value=0, value=100, step=10)

    st.divider()
    st.markdown("**Advanced (English)**")
    lookback_days = st.slider("Indicators lookback (days)", 60, 400, 200, 20)
    max_symbols = st.slider("Max symbols to analyze", 10, 200, 60, 10)
    w_price_vs_sma50 = st.number_input("Weight: Price above SMA50", value=2.0, step=0.5)
    w_trend_sma20_gt_sma50 = st.number_input("Weight: SMA20 > SMA50", value=1.0, step=0.5)
    w_mom_20d = st.number_input("Weight: 20d momentum", value=1.0, step=0.5)
    w_rsi_band = st.number_input("Weight: RSI 50–70 band", value=1.5, step=0.5)
    w_vol_penalty = st.number_input("Weight: Volatility penalty", value=1.0, step=0.5)

# -------------------------------
# Helpers
# -------------------------------
@st.cache_data(show_spinner=False)
def fetch_kospi200_symbols_from_wikipedia() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/KOSPI_200"
    tables = pd.read_html(url)
    candidates = [t for t in tables if {"Company", "Symbol"}.issubset(set(t.columns))]
    if not candidates:
        raise RuntimeError("KOSPI 200 table not found.")
    df = candidates[0].copy()
    df["Symbol"] = df["Symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    df = df.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
    df = df.rename(columns={"Company": "name", "Symbol": "symbol"})
    return df[["symbol", "name"]]

@st.cache_data(show_spinner=False)
def fallback_symbols() -> pd.DataFrame:
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
    return {"Authorization": f"apikey {TD_API_KEY}"} if TD_API_KEY else {}

def td_get(path: str, params: dict) -> dict:
    params = dict(params or {})
    if TD_API_KEY and "apikey" not in params:
        params["apikey"] = TD_API_KEY
    url = f"{TD_BASE.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, params=params, headers=td_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()
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
    out_size = max(lookback_days + 50, 150)
    params = {
        "symbol": symbol,     # 6-digit KRX code, e.g., 005930
        "interval": "1day",
        "outputsize": out_size,
        "format": "JSON",
    }
    data = td_get("/time_series", params)
    if "values" not in data:
        raise RuntimeError(f"No time_series values for {symbol}: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def score_symbol(ts: pd.DataFrame) -> dict:
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

    vol20 = close.pct_change().rolling(20).std().iloc[-1]

    _sma20 = float(sma20.iloc[-1])
    _sma50 = float(sma50.iloc[-1])
    _rsi14 = float(rsi14.iloc[-1])

    s = 0.0
    s += w_price_vs_sma50 if latest_close > _sma50 else 0.0
    s += w_trend_sma20_gt_sma50 if _sma20 > _sma50 else 0.0
    if not np.isnan(mom_20d):
        s += w_mom_20d * (mom_20d * 100)
    if 50 <= _rsi14 <= 70:
        s += w_rsi_band
    elif _rsi14 > 75:
        s -= 0.5
    elif _rsi14 < 30:
        s -= 0.5
    if not np.isnan(vol20):
        s -= w_vol_penalty * (vol20 * 100)

    # --- Liquidity proxy (for "최소 거래 규모 (억원)") ---
    # Use 20-day average trading value = mean(close * volume).
    # KRX prices are KRW; volume is shares. Convert to 억원 by dividing 1e8.
    value20 = (ts["close"] * ts["volume"]).rolling(20).mean().iloc[-1]
    ogwon20 = float(value20) / 1e8 if pd.notna(value20) else np.nan

    return {
        "date": latest_date.date().isoformat(),
        "close": latest_close,
        "sma20": _sma20,
        "sma50": _sma50,
        "rsi14": _rsi14,
        "mom_20d_pct": mom_20d * 100 if not np.isnan(mom_20d) else np.nan,
        "vol20_pct": vol20 * 100 if not np.isnan(vol20) else np.nan,
        "avg20_trading_ogwon": ogwon20,
        "score": s,
    }

# -------------------------------
# Universe
# -------------------------------
st.subheader("Universe: KOSPI 200 components (auto-fetched)")
try:
    kospi_df = fetch_kospi200_symbols_from_wikipedia()
    st.success(f"Fetched {len(kospi_df)} symbols from Wikipedia.")
except Exception as e:
    st.warning(f"Auto-fetch failed ({e}). Using fallback sample list.")
    kospi_df = fallback_symbols()

universe = kospi_df.head(max_symbols).copy()
st.dataframe(universe.head(20), use_container_width=True)

if not TD_API_KEY:
    st.stop()

# -------------------------------
# Run
# -------------------------------
st.subheader("Run analysis")
if st.button("Analyze & Rank"):
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
            rows.append({"symbol": sym, "name": name, **metrics})
        except Exception:
            time.sleep(0.2)
        finally:
            progress.progress(int((i + 1) / len(universe) * 100))

    if not rows:
        st.error("No results. Adjust limits or lookback.")
        st.stop()

    results = pd.DataFrame(rows)

    # --- Apply liquidity filter by "최소 거래 규모 (억원)" ---
    mask_liq = results["avg20_trading_ogwon"].fillna(0) >= float(min_trading_ogwon)
    filtered = results[mask_liq].copy()

    if filtered.empty:
        st.warning("No symbols meet the minimum trading size. Lower the threshold or expand universe.")
        filtered = results.copy()

    filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)

    st.success(f"Computed {len(results)} symbols; {mask_liq.sum()} passed liquidity filter (최소 거래 규모).")
    st.subheader(f"Top {top_k} Recommendations")
    st.dataframe(
        filtered.loc[:, ["symbol", "name", "date", "close", "sma20", "sma50",
                         "rsi14", "mom_20d_pct", "vol20_pct", "avg20_trading_ogwon", "score"]].head(top_k),
        use_container_width=True
    )

    pick = st.selectbox("Select a symbol to visualize", filtered["symbol"].head(top_k).tolist())
    if pick:
        detail_row = filtered[filtered["symbol"] == pick].iloc[0]
        st.markdown(f"**{detail_row['symbol']} — {detail_row['name']}**  (Last date: {detail_row['date']})")
        st.write({
            "close": round(detail_row["close"], 2),
            "SMA20": round(detail_row["sma20"], 2),
            "SMA50": round(detail_row["sma50"], 2),
            "RSI(14)": round(detail_row["rsi14"], 2),
            "20d momentum %": round(detail_row["mom_20d_pct"], 2) if pd.notna(detail_row["mom_20d_pct"]) else None,
            "20d vol % (stdev)": round(detail_row["vol20_pct"], 2) if pd.notna(detail_row["vol20_pct"]) else None,
            "Avg 20d trading (억원)": round(detail_row["avg20_trading_ogwon"], 2) if pd.notna(detail_row["avg20_trading_ogwon"]) else None,
            "score": round(detail_row["score"], 2),
        })

        # Quick chart (Altair)
        try:
            ts = fetch_daily_timeseries(pick, lookback_days=lookback_days).tail(lookback_days).copy()
            ts["SMA20"] = ts["close"].rolling(20).mean()
            ts["SMA50"] = ts["close"].rolling(50).mean()

            import altair as alt
            base = alt.Chart(ts).encode(x="datetime:T")
            price_line = base.mark_line().encode(y=alt.Y("close:Q", title="Price"))
            sma20_line = base.mark_line(strokeDash=[4,2]).encode(y="SMA20:Q")
            sma50_line = base.mark_line(strokeDash=[2,2]).encode(y="SMA50:Q")
            st.altair_chart((price_line + sma20_line + sma50_line).properties(height=320), use_container_width=True)
        except Exception as e:
            st.warning(f"Charting failed: {e}")

    st.caption("Heuristic scoring. Tune weights and thresholds to explore different regimes.")
