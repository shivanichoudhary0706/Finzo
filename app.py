"""
Finzo — live stock-ranking dashboard
Run:  streamlit run app.py
"""
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from xgboost import XGBRegressor

st.set_page_config(page_title="Finzo", page_icon="📈", layout="wide")

# ───────────────────────────── Settings ─────────────────────────────
SECTOR = {'RELIANCE.NS': 'Energy', 'TCS.NS': 'IT', 'INFY.NS': 'IT', 'WIPRO.NS': 'IT', 'HCLTECH.NS': 'IT',
          'HDFCBANK.NS': 'Bank', 'ICICIBANK.NS': 'Bank', 'SBIN.NS': 'Bank', 'KOTAKBANK.NS': 'Bank',
          'AXISBANK.NS': 'Bank', 'BAJFINANCE.NS': 'Finance', 'HINDUNILVR.NS': 'FMCG', 'ITC.NS': 'FMCG',
          'NESTLEIND.NS': 'FMCG', 'BHARTIARTL.NS': 'Telecom', 'LT.NS': 'Infra', 'ASIANPAINT.NS': 'Consumer',
          'TITAN.NS': 'Consumer', 'MARUTI.NS': 'Auto', 'M&M.NS': 'Auto', 'SUNPHARMA.NS': 'Pharma',
          'ULTRACEMCO.NS': 'Cement', 'POWERGRID.NS': 'Power', 'NTPC.NS': 'Power', 'TATASTEEL.NS': 'Metals'}
TICKERS = list(SECTOR)
START, HORIZON, TOP_N, MAX_PER_SECTOR, COST = '2015-01-01', 20, 5, 2, 0.002   # COST = round trip
FEATURES = ['ret_1', 'ret_5', 'ret_20', 'ret_60', 'dist_sma20', 'dist_sma50', 'dist_sma200',
            'sma20_vs_50', 'rsi_14', 'macd_hist', 'bb_pctb', 'vol_20', 'atr_pct', 'volume_ratio']
IST = ZoneInfo('Asia/Kolkata')

# Chart colours (validated categorical palette)
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#b5b3ab'
GOOD, BAD = '#0ca30c', '#d03b3b'
GRID = 'rgba(137,135,129,0.25)'


def short(t):
    return t.replace('.NS', '')


def style(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode='x unified',
                      legend=dict(orientation='h', y=1.08, x=0), plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ───────────────────────────── Data & features ─────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)       # re-fetch prices every 15 minutes
def fetch_all():
    raw = {}
    for t in TICKERS:
        try:
            d = yf.download(t, start=START, auto_adjust=True, multi_level_index=False, progress=False)
            if len(d) > 300:
                raw[t] = d
        except Exception:
            pass
    return raw, datetime.now(IST)


def add_features(d):
    d = d.copy()
    c = d['Close']
    d['ret_1'], d['ret_5'] = c.pct_change(), c.pct_change(5)
    d['ret_20'], d['ret_60'] = c.pct_change(20), c.pct_change(60)
    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    d['SMA20'], d['SMA50'] = sma20, sma50
    d['dist_sma20'], d['dist_sma50'], d['dist_sma200'] = c/sma20 - 1, c/sma50 - 1, c/sma200 - 1
    d['sma20_vs_50'] = sma20/sma50 - 1
    delta = c.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d['rsi_14'] = 100 - 100 / (1 + avg_gain / avg_loss)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d['macd_hist'] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    std20 = c.rolling(20).std()
    d['bb_pctb'] = (c - (sma20 - 2*std20)) / (4*std20)
    d['vol_20'] = d['ret_1'].rolling(20).std() * np.sqrt(252)
    prev = c.shift(1)
    tr = pd.concat([d['High'] - d['Low'], (d['High'] - prev).abs(), (d['Low'] - prev).abs()], axis=1).max(axis=1)
    d['atr_pct'] = tr.rolling(14).mean() / c
    d['volume_ratio'] = d['Volume'] / d['Volume'].rolling(20).mean()
    d['fwd_ret'] = c.shift(-HORIZON) / c - 1
    return d.replace([np.inf, -np.inf], np.nan)


def xgb():
    return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)


def pick_top(day, score_col, n=TOP_N, cap=MAX_PER_SECTOR):
    chosen, count = [], {}
    for _, r in day.sort_values(score_col, ascending=False).iterrows():
        s = SECTOR.get(r['Ticker'], 'Other')
        if count.get(s, 0) < cap:
            chosen.append(r)
            count[s] = count.get(s, 0) + 1
        if len(chosen) == n:
            break
    return pd.DataFrame(chosen)


@st.cache_resource(ttl=86400, show_spinner=False)  # retrain at most once a day
def train_final(_labeled, version):
    return xgb().fit(_labeled[FEATURES], _labeled['fwd_ret'])


@st.cache_data(ttl=86400, show_spinner=False)
def walk_forward(_labeled, version):
    years, periods = [], []
    last_year = _labeled['Date'].dt.year.max()
    for year in range(2019, last_year + 1):
        tr = _labeled[_labeled['Date'] < f'{year-1}-12-01']
        te = _labeled[_labeled['Date'].dt.year == year]
        if len(te) == 0 or len(tr) == 0:
            continue
        te = te.assign(pred=xgb().fit(tr[FEATURES], tr['fwd_ret']).predict(te[FEATURES]))
        ic = te.groupby('Date').apply(lambda g: spearmanr(g['pred'], g['fwd_ret'])[0]).mean()
        edges = []
        for dt in sorted(te['Date'].unique())[::HORIZON]:
            day = te[te['Date'] == dt]
            p, a = pick_top(day, 'pred')['fwd_ret'].mean(), day['fwd_ret'].mean()
            periods.append({'Date': dt, 'Finzo picks': p - COST, 'All 25 stocks': a})
            edges.append(p - a)
        years.append({'Year': year, 'IC': ic, 'Edge %': np.mean(edges) * 100})
    return pd.DataFrame(years), pd.DataFrame(periods).set_index('Date')


# ───────────────────────────── Header ─────────────────────────────
now = datetime.now(IST)
market_open = now.weekday() < 5 and dtime(9, 15) <= now.time() <= dtime(15, 30)

head_l, head_r = st.columns([3, 1])
with head_l:
    st.title("📈 Finzo")
    st.caption("Ranks 25 large NSE stocks by their predicted return over the next 20 trading days (~1 month).")
with head_r:
    auto = st.toggle("Auto-refresh every 15 min", value=True)
    if st.button("🔄 Refresh now", width='stretch'):
        st.cache_data.clear()
        st.rerun()
if auto:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=15 * 60 * 1000, key="finzo_refresh")
    except ImportError:
        pass

with st.spinner("Fetching prices and running the model… (first load takes about a minute)"):
    raw, fetched_at = fetch_all()
    if not raw:
        st.error("Couldn't download prices from Yahoo Finance. Try **Refresh now** in a minute.")
        st.stop()
    feat = {t: add_features(d) for t, d in raw.items()}
    panel = pd.concat([d.assign(Ticker=t) for t, d in feat.items()])
    panel.index.name = 'Date'
    panel = panel.reset_index()
    labeled = panel.dropna(subset=FEATURES + ['fwd_ret'])
    version = str(labeled['Date'].max().date())
    model = train_final(labeled, version)
    wf_years, wf_periods = walk_forward(labeled, version)

latest = panel.dropna(subset=FEATURES).sort_values('Date').groupby('Ticker').tail(1).copy()
latest['pred'] = model.predict(latest[FEATURES])
latest['Sector'] = latest['Ticker'].map(SECTOR)
prev_close = {t: d['Close'].iloc[-2] for t, d in raw.items()}
latest['Day %'] = [(c / prev_close[t] - 1) * 100 for t, c in zip(latest['Ticker'], latest['Close'])]
latest = latest.sort_values('pred', ascending=False)
latest['Rank'] = range(1, len(latest) + 1)
picks = pick_top(latest, 'pred')
data_date = latest['Date'].max().date()

status = "🟢 Market open — prices update during the day (Yahoo data is ~15 min delayed)" if market_open \
    else "⚪ Market closed — showing the latest closing prices"
st.info(f"{status}  ·  Price data as of **{data_date:%d %b %Y}**  ·  Last fetched **{fetched_at:%d %b, %I:%M %p} IST**")

# KPI row
edge_after = wf_years['Edge %'].mean() - COST * 100
k1, k2, k3, k4 = st.columns(4)
k1.metric("Model edge vs all 25 stocks", f"{edge_after:+.2f}% / month",
          help="Average extra return of Finzo's top 5 over buying all 25 equally, per 20-day period, "
               "after ~0.2% trading cost. Measured with walk-forward testing.")
k2.metric("Years it beat the market", f"{(wf_years['Edge %'] > 0).sum()} of {len(wf_years)}")
k3.metric("Ranking skill (IC)", f"{wf_years['IC'].mean():.3f}",
          help="Correlation between predicted and actual ranking. 0 = no skill, 0.03–0.05 = useful.")
k4.metric("Stocks tracked", f"{len(raw)}")

tab_picks, tab_explore, tab_health, tab_how = st.tabs(
    ["🏆 Today's picks", "📈 Stock explorer", "🧪 Model health", "📚 How Finzo works"])

# ───────────────────────────── Tab 1: Picks ─────────────────────────────
with tab_picks:
    st.subheader(f"Top {TOP_N} for the next 20 trading days")
    st.caption(f"Maximum {MAX_PER_SECTOR} stocks per sector, so the picks aren't one big bet on a single industry.")
    cols = st.columns(TOP_N)
    for col, (_, r) in zip(cols, picks.iterrows()):
        with col, st.container(border=True):
            st.markdown(f"**#{int(r['Rank'])} · {short(r['Ticker'])}**  \n{r['Sector']}")
            st.metric("Price", f"₹{r['Close']:,.1f}", f"{r['Day %']:+.2f}% today")
            st.markdown(f"Expected 20d: **{r['pred']*100:+.2f}%**")
            zone = "🔥 Overbought" if r['rsi_14'] > 70 else "🧊 Oversold" if r['rsi_14'] < 30 else "⚖️ Neutral"
            trend = "📈 Uptrend" if r['Close'] > r['SMA50'] else "📉 Downtrend"
            st.caption(f"RSI {r['rsi_14']:.0f} {zone}  \n{trend} (vs SMA 50)")

    st.markdown("#### Full ranking")
    rank = latest.iloc[::-1]
    is_pick = rank['Ticker'].isin(picks['Ticker'])
    fig = go.Figure(go.Bar(
        x=rank['pred'] * 100, y=rank['Ticker'].map(short), orientation='h',
        marker=dict(color=np.where(is_pick, BLUE, GRAY), cornerradius=4),
        customdata=np.stack([rank['Sector'], rank['Close'], rank['rsi_14']], axis=1),
        hovertemplate='<b>%{y}</b> (%{customdata[0]})<br>Expected 20d: %{x:.2f}%'
                      '<br>Price: ₹%{customdata[1]:,.1f}<br>RSI: %{customdata[2]:.0f}<extra></extra>'))
    fig.update_layout(title="Expected 20-day return — blue = Finzo's picks", xaxis_ticksuffix='%')
    fig.update_layout(hovermode='closest')
    st.plotly_chart(style(fig, 640), width='stretch')

    st.caption("💡 The expected return is a **ranking score**, not a promise. The model gets the *order* of "
               "stocks slightly right more often than chance; any single stock can still move ±5% or more.")
    with st.expander("Show as table"):
        tbl = latest[['Rank', 'Ticker', 'Sector', 'Close', 'Day %', 'rsi_14', 'pred']].copy()
        tbl['Ticker'] = tbl['Ticker'].map(short)
        tbl.columns = ['Rank', 'Stock', 'Sector', 'Price ₹', 'Today %', 'RSI', 'Expected 20d %']
        tbl['Expected 20d %'] *= 100
        st.dataframe(tbl.round(2), hide_index=True, width='stretch')

# ───────────────────────────── Tab 2: Explorer ─────────────────────────────
with tab_explore:
    c1, c2 = st.columns([2, 1])
    options = list(latest['Ticker'])
    ticker = c1.selectbox("Stock", options, format_func=lambda t: f"{short(t)} — {SECTOR[t]}")
    period = c2.radio("Period", ['6M', '1Y', '3Y', '5Y'], index=1, horizontal=True)
    days = {'6M': 126, '1Y': 252, '3Y': 756, '5Y': 1260}[period]
    d = feat[ticker].tail(days)
    row = latest[latest['Ticker'] == ticker].iloc[0]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06,
                        subplot_titles=(f"{short(ticker)} — price & moving averages", "RSI (14)"))
    fig.add_trace(go.Scatter(x=d.index, y=d['Close'], name='Price', line=dict(color=BLUE, width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d['SMA20'], name='SMA 20', line=dict(color=ORANGE, width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d['SMA50'], name='SMA 50', line=dict(color=AQUA, width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d['rsi_14'], name='RSI', line=dict(color=BLUE, width=2),
                             showlegend=False), 2, 1)
    for lvl, txt in [(70, 'Overbought 70'), (30, 'Oversold 30')]:
        fig.add_hline(y=lvl, line=dict(color='#898781', dash='dot', width=1), row=2, col=1,
                      annotation_text=txt, annotation_position='right', annotation_font_size=10)
    fig.update_yaxes(tickprefix='₹', row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    st.plotly_chart(style(fig, 560), width='stretch')

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Finzo rank", f"#{int(row['Rank'])} of {len(latest)}")
    m2.metric("Expected 20d", f"{row['pred']*100:+.2f}%")
    m3.metric("RSI", f"{row['rsi_14']:.0f}")
    m4.metric("1-month change", f"{row['ret_20']*100:+.1f}%")

    above50 = row['Close'] > row['SMA50']
    cross = "above" if row['SMA20'] > row['SMA50'] else "below"
    rsi_txt = ("overbought — buyers have dominated recently" if row['rsi_14'] > 70 else
               "oversold — sellers have dominated recently" if row['rsi_14'] < 30 else
               "neutral — no extreme buying or selling")
    st.markdown(f"""
**What the indicators say right now**
- **Trend:** price is {'above' if above50 else 'below'} its 50-day average → {'📈 uptrend' if above50 else '📉 downtrend'}.
- **Moving-average cross:** SMA 20 is {cross} SMA 50 ({'golden cross zone' if cross == 'above' else 'death cross zone'}).
- **RSI {row['rsi_14']:.0f}:** {rsi_txt}.
""")

# ───────────────────────────── Tab 3: Model health ─────────────────────────────
with tab_health:
    st.subheader("Does Finzo actually work? Walk-forward test")
    st.caption("Each year, the model is trained only on earlier years and then tested on that year, "
               "like 8 separate exams. Picks rebalance every 20 trading days.")
    h1, h2 = st.columns(2)
    with h1:
        fig = go.Figure(go.Bar(
            x=wf_years['Year'].astype(str), y=wf_years['Edge %'],
            marker=dict(color=np.where(wf_years['Edge %'] >= 0, BLUE, ORANGE), cornerradius=4),
            hovertemplate='%{x}: %{y:+.2f}% per 20 days<extra></extra>'))
        fig.update_layout(title="Extra return vs all 25 stocks, per 20 days (before costs)",
                          yaxis_ticksuffix='%', hovermode='closest', xaxis_type='category')
        st.plotly_chart(style(fig, 380), width='stretch')
    with h2:
        eq = 100000 * (1 + wf_periods).cumprod()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq['Finzo picks'], name='Finzo top 5 (after costs)',
                                 line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=eq.index, y=eq['All 25 stocks'], name='All 25 stocks equally',
                                 line=dict(color=GRAY, width=2)))
        fig.update_layout(title="₹1,00,000 followed through every test year", yaxis_tickprefix='₹')
        st.plotly_chart(style(fig, 380), width='stretch')
    show = wf_years.assign(IC=wf_years['IC'].round(3), **{'Edge %': wf_years['Edge %'].round(2)})
    show['Beat the market?'] = np.where(show['Edge %'] > 0, '✅', '❌')
    st.dataframe(show, hide_index=True, width='stretch')
    st.warning("**Limitations:** the 25 stocks are today's large companies (survivorship bias flatters results); "
               "the edge is small and uneven between years; past results don't guarantee future ones.")

# ───────────────────────────── Tab 4: How it works ─────────────────────────────
with tab_how:
    st.subheader("How Finzo works")
    st.markdown("""
| Step | What happens |
|---|---|
| **1. Fetch** | Daily prices for 25 large NSE stocks since 2015 from Yahoo Finance (refreshed every 15 min) |
| **2. Indicators** | 14 features: momentum (1d/5d/20d/60d returns), distance from SMA 20/50/200, golden-cross strength, RSI, MACD, Bollinger %B, volatility, ATR, volume surge |
| **3. Learn** | An XGBoost model learns how those features relate to the next 20 days' return (retrained daily) |
| **4. Rank** | Every stock gets a score; Finzo picks the top 5, max 2 per sector |
| **5. Verify** | Walk-forward testing checks the ranking year by year on data the model never saw |

**What we tried and rejected (with evidence):**
- SMA 20/50 crossover trading rule → lost heavily to buy & hold on Reliance
- "Will it go up?" classifiers → no better than always guessing "up"
- Relative-return target → less consistent than predicting plain returns

⚠️ **Finzo is a learning project, not investment advice.** It ranks stocks; it doesn't know about news, results or events.
""")