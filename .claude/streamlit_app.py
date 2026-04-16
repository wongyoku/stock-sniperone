"""
股票量化精选系统 - Streamlit 可视化版
======================================
从沪深A股全市场筛选优质股票
- 技术面筛选：MA20趋势 + KDJ择时 + 量价配合
- 可视化展示：K线图、技术指标、评分排行

Author: AI量化助手
"""

import requests
import os
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]

original_session = requests.Session
def patched_session(*args, **kwargs):
    session = original_session(*args, **kwargs)
    session.trust_env = False
    return session
requests.Session = patched_session

import streamlit as st
import akshare as ak
import pandas as pd
import datetime
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ================= 配置参数 =================

SCAN_CONFIG = {
    "max_stocks": 500,
    "min_turnover": 1e8,
}

TECHNICAL_PARAMS = {
    "ma_period": 20,
    "kdj_n": 9,
    "kdj_m1": 3,
    "kdj_m2": 3,
    "volume_ratio_min": 1.3,
    "price_change_max": 0.09,
}

FUNDAMENTAL_FILTERS = {
    "enabled": False,
    "pe_min": 5,
    "pe_max": 80,
    "market_cap_min": 50,
    "market_cap_max": 5000,
}

SIGNAL_WEIGHTS = {
    "technical_score": 0.6,
    "momentum_score": 0.4,
}


# ================= 数据获取模块 =================

@st.cache_data(ttl=3600)
def get_all_stocks():
    """获取沪深A股全市场股票列表"""
    try:
        df = ak.stock_zh_a_spot()
        df = df.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10]]
        df.columns = ['代码', '名称', '最新价', '涨跌幅', '最高', '最低', '成交量', '成交额']

        for col in ['最新价', '涨跌幅', '最高', '最低', '成交量', '成交额']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['名称'] = df['名称'].fillna('').astype(str)
        df = df[~df['名称'].str.contains('ST|退市', na=False, case=False)]
        df = df[df['涨跌幅'] > -9.9]
        df = df.sort_values('成交额', ascending=False)
        df = df.head(SCAN_CONFIG["max_stocks"])

        return df
    except Exception as e:
        st.error(f"获取股票列表失败: {e}")
        return None


@st.cache_data(ttl=3600)
def get_stock_hist_data(symbol):
    """获取个股历史数据"""
    try:
        df = ak.stock_zh_a_daily(symbol=symbol, adjust='qfq')

        if not isinstance(df, pd.DataFrame) or df is None:
            return None
        if len(df) < 30:
            return None

        col_rename = {}
        for col in df.columns:
            col_lower = col.lower().strip() if isinstance(col, str) else str(col).lower()
            if col_lower in ['date', '日期', '时间', 'datetime']:
                col_rename[col] = 'date'
            elif col_lower in ['open', '开盘', '开盘价']:
                col_rename[col] = 'open'
            elif col_lower in ['high', '最高', '最高价']:
                col_rename[col] = 'high'
            elif col_lower in ['low', '最低', '最低价']:
                col_rename[col] = 'low'
            elif col_lower in ['close', '收盘', '收盘价']:
                col_rename[col] = 'close'
            elif col_lower in ['volume', '成交量']:
                col_rename[col] = 'volume'

        df = df.rename(columns=col_rename)

        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return None

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').tail(60).copy()

        return df
    except Exception:
        return None


# ================= 技术分析模块 =================

def calculate_indicators(df):
    """计算技术指标"""
    if df is None or len(df) < 30:
        return None

    df.columns = [col.strip() for col in df.columns]
    col_map = {
        '日期': 'date', '开盘': 'open', '最高': 'high',
        '最低': 'low', '收盘': 'close', '成交量': 'volume',
        '成交额': 'amount'
    }
    df = df.rename(columns=col_map)

    ma_period = TECHNICAL_PARAMS["ma_period"]
    df['MA'] = df['close'].rolling(window=ma_period).mean()

    n = TECHNICAL_PARAMS["kdj_n"]
    m1 = TECHNICAL_PARAMS["kdj_m1"]
    m2 = TECHNICAL_PARAMS["kdj_m2"]

    low_list = df['low'].rolling(window=n, min_periods=1).min()
    high_list = df['high'].rolling(window=n, min_periods=1).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100

    df['K'] = rsv.ewm(com=m1-1, adjust=False).mean()
    df['D'] = df['K'].ewm(com=m2-1, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    df['price_change'] = df['close'].pct_change()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['ma_distance'] = (df['close'] - df['MA']) / df['MA']
    df['MA偏离度'] = ((df['close'] - df['MA']) / df['MA'] * 100).round(2)

    return df.dropna().reset_index(drop=True)


def calculate_technical_score(df):
    """技术面评分 (0-100)"""
    if df is None or len(df) < 5:
        return 0

    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0

    if last['close'] > last['MA']:
        score += 30
        ma_score = min(10, last['ma_distance'] * 100)
        score += ma_score

    if last['J'] < 20:
        score += 25
    elif last['J'] < 40:
        score += 15

    if last['J'] > last['K'] and prev['J'] <= prev['K'] and last['J'] < 60:
        score += 20

    if last['volume_ratio'] > TECHNICAL_PARAMS["volume_ratio_min"]:
        score += 15

    change = last['close'] / df.iloc[-2]['close'] - 1 if len(df) > 1 else 0
    if 0 < change < TECHNICAL_PARAMS["price_change_max"]:
        score += 10
    elif change >= TECHNICAL_PARAMS["price_change_max"]:
        score += 3

    return min(100, score)


def evaluate_buy_signal(df):
    """评估买入信号"""
    if df is None or len(df) < 5:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    trend_up = last['close'] > last['MA']
    kdj_oversold = last['J'] < 30
    kdj_golden_cross = last['J'] > last['K'] and prev['J'] <= prev['K']
    kdj_bounce = kdj_oversold and (last['J'] > prev['J'])
    volume_ok = last['volume_ratio'] > TECHNICAL_PARAMS["volume_ratio_min"]

    if trend_up and (kdj_golden_cross or kdj_bounce):
        signal = "[强买入] 上升趋势+KDJ信号共振"
        priority = 1
    elif trend_up and kdj_oversold:
        signal = "[中买入] 趋势向上+KDJ超卖蓄力"
        priority = 2
    elif kdj_golden_cross and volume_ok:
        signal = "[轻买入] KDJ金叉+放量，需确认趋势"
        priority = 3
    elif last['J'] > 85:
        signal = "[观望] KDJ超买"
        priority = 7
    elif not trend_up:
        signal = "[观望] 趋势向下"
        priority = 7
    elif kdj_oversold:
        signal = "[关注] KDJ超卖反弹预期"
        priority = 4
    else:
        signal = "[观望] 信号不明显"
        priority = 6

    return {
        "signal": signal,
        "priority": priority,
        "trend_up": trend_up,
        "kdj_oversold": kdj_oversold,
        "kdj_golden_cross": kdj_golden_cross,
        "volume_ok": volume_ok,
        "J值": round(last['J'], 1),
        "MA偏离度": last['MA偏离度'],
        "量比": round(last['volume_ratio'], 2),
        "涨幅": round((last['close'] / df.iloc[-2]['close'] - 1) * 100, 2),
    }


# ================= K线图绘制 =================

def plot_stock_chart(df, stock_name, stock_code):
    """绘制个股K线图+MA+KDJ"""
    if df is None or len(df) < 20:
        return None

    # 计算指标
    ma_period = TECHNICAL_PARAMS["ma_period"]
    n = TECHNICAL_PARAMS["kdj_n"]
    m1 = TECHNICAL_PARAMS["kdj_m1"]
    m2 = TECHNICAL_PARAMS["kdj_m2"]

    df['MA'] = df['close'].rolling(window=ma_period).mean()

    low_list = df['low'].rolling(window=n, min_periods=1).min()
    high_list = df['high'].rolling(window=n, min_periods=1).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['K'] = rsv.ewm(com=m1-1, adjust=False).mean()
    df['D'] = df['K'].ewm(com=m2-1, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # 创建子图
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.2, 0.2],
        subplot_titles=(f'{stock_name}({stock_code}) K线图', '成交量', 'KDJ指标')
    )

    # K线
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='#FF4136',
            decreasing_line_color='#3D9970'
        ),
        row=1, col=1
    )

    # MA线
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['MA'], name=f'MA{ma_period}',
                   line=dict(color='#FF851B', width=1.5)),
        row=1, col=1
    )

    # 成交量
    colors = ['#FF4136' if df['close'].iloc[i] >= df['open'].iloc[i] else '#3D9970'
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], name='成交量', marker_color=colors),
        row=2, col=1
    )

    # KDJ
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['K'], name='K', line=dict(color='#0074D9', width=1)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['D'], name='D', line=dict(color='#B10DC9', width=1)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['J'], name='J', line=dict(color='#FF851B', width=1)),
        row=3, col=1
    )

    # 超买超卖线
    fig.add_hline(y=80, line_dash="dash", line_color="#FF4136", row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="#3D9970", row=3, col=1)

    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    return fig


# ================= 扫描市场 =================

def scan_market():
    """扫描全市场并筛选标的"""
    all_stocks = get_all_stocks()
    if all_stocks is None or len(all_stocks) == 0:
        return None

    candidates = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (_, stock) in enumerate(all_stocks.iterrows()):
        code = stock['代码']
        name = stock['名称']

        # 更新进度
        progress = (idx + 1) / len(all_stocks)
        progress_bar.progress(progress)
        status_text.text(f"正在扫描 {code} {name}... ({idx+1}/{len(all_stocks)})")

        hist_df = get_stock_hist_data(code)
        if hist_df is None:
            continue

        tech_df = calculate_indicators(hist_df)
        if tech_df is None or len(tech_df) < 5:
            continue

        buy_signal = evaluate_buy_signal(tech_df)
        if buy_signal is None or buy_signal['priority'] >= 7:
            continue

        tech_score = calculate_technical_score(tech_df)

        if len(tech_df) >= 5:
            momentum = (tech_df.iloc[-1]['close'] / tech_df.iloc[-5]['close'] - 1) * 100
        else:
            momentum = 0

        total_score = tech_score * SIGNAL_WEIGHTS["technical_score"] + \
                      min(100, momentum + 50) * SIGNAL_WEIGHTS["momentum_score"]

        candidate = {
            "代码": code,
            "名称": name,
            "最新价": stock['最新价'],
            "涨跌幅": stock['涨跌幅'],
            "技术评分": round(tech_score, 1),
            "综合评分": round(total_score, 1),
            "J值": buy_signal['J值'],
            "MA偏离度": buy_signal['MA偏离度'],
            "量比": buy_signal['量比'],
            "信号": buy_signal['signal'],
            "优先级": buy_signal['priority'],
        }
        candidates.append(candidate)

    progress_bar.progress(100)
    status_text.text("扫描完成！")
    return pd.DataFrame(candidates)


# ================= Streamlit 页面 =================

def main():
    st.set_page_config(
        page_title="股票量化精选系统",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 侧边栏配置
    st.sidebar.header("参数设置")

    SCAN_CONFIG["max_stocks"] = st.sidebar.slider(
        "扫描股票数量", 100, 1000, 500, 50
    )

    TECHNICAL_PARAMS["ma_period"] = st.sidebar.slider(
        "MA周期", 5, 60, 20, 5
    )

    TECHNICAL_PARAMS["volume_ratio_min"] = st.sidebar.slider(
        "最小量比", 1.0, 3.0, 1.3, 0.1
    )

    TECHNICAL_PARAMS["price_change_max"] = st.sidebar.slider(
        "最大涨幅限制", 0.05, 0.15, 0.09, 0.01
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 技术指标参数")
    TECHNICAL_PARAMS["kdj_n"] = st.sidebar.slider("KDJ N", 3, 21, 9)
    TECHNICAL_PARAMS["kdj_m1"] = st.sidebar.slider("KDJ M1", 2, 10, 3)
    TECHNICAL_PARAMS["kdj_m2"] = st.sidebar.slider("KDJ M2", 2, 10, 3)

    # 主页面
    st.title("📈 股票量化精选系统")
    st.markdown(f"**扫描日期:** {datetime.date.today()} | **扫描范围:** 沪深A股全市场")

    # 扫描按钮
    col1, col2 = st.columns([1, 3])
    with col1:
        scan_button = st.button("🔍 开始扫描市场", type="primary", use_container_width=True)

    if scan_button or 'results' in st.session_state:
        if scan_button:
            with st.spinner("正在扫描市场，请稍候..."):
                results = scan_market()
                st.session_state['results'] = results
                st.session_state['scan_time'] = datetime.datetime.now()
        else:
            results = st.session_state.get('results')

        if results is None or len(results) == 0:
            st.warning("今日无符合条件的标的")
            return

        results = results.sort_values(['优先级', '综合评分'], ascending=[True, False])

        # 概览统计
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("扫描股票总数", f"{SCAN_CONFIG['max_stocks']} 只")
        with col2:
            st.metric("符合条件", f"{len(results)} 只")
        with col3:
            strong_buy = len(results[results['优先级'] == 1])
            st.metric("强买入信号", f"{strong_buy} 只", delta=None, delta_color="normal")
        with col4:
            avg_score = results['综合评分'].mean()
            st.metric("平均评分", f"{avg_score:.1f}")

        # TOP推荐
        st.markdown("---")
        st.subheader("⭐ TOP推荐")

        tab1, tab2 = st.tabs(["优先级1 - 强烈买入", "优先级2 - 较好标的"])

        with tab1:
            top1 = results[results['优先级'] == 1].head(10)
            if len(top1) > 0:
                st.dataframe(
                    top1[['代码', '名称', '最新价', '涨跌幅', '综合评分', 'J值', 'MA偏离度', '信号']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("今日无优先级1信号")

        with tab2:
            top2 = results[results['优先级'] == 2].head(10)
            if len(top2) > 0:
                st.dataframe(
                    top2[['代码', '名称', '最新价', '涨跌幅', '综合评分', 'J值', 'MA偏离度', '信号']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("今日无优先级2信号")

        # 完整排行榜
        st.markdown("---")
        st.subheader("📊 完整评分榜 TOP 30")

        display_df = results.head(30)[['代码', '名称', '最新价', '涨跌幅', '综合评分', 'J值', 'MA偏离度', '量比', '信号']].copy()
        display_df['涨跌幅'] = display_df['涨跌幅'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # 个股详情查看
        st.markdown("---")
        st.subheader("🔍 个股详情查看")

        selected_stock = st.selectbox(
            "选择股票查看详细K线图",
            options=results['代码'].tolist(),
            format_func=lambda x: f"{x} {results[results['代码']==x]['名称'].values[0]}"
        )

        if selected_stock:
            stock_info = results[results['代码'] == selected_stock].iloc[0]
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("最新价", f"¥{stock_info['最新价']:.2f}")
            with col2:
                st.metric("涨跌幅", f"{stock_info['涨跌幅']:.2f}%",
                          delta=stock_info['涨跌幅'], delta_color="normal")
            with col3:
                st.metric("综合评分", f"{stock_info['综合评分']}")
            with col4:
                st.metric("J值", f"{stock_info['J值']}")
            with col5:
                st.metric("MA偏离度", f"{stock_info['MA偏离度']}%")

            st.markdown(f"**信号:** {stock_info['信号']}")

            # 绘制K线图
            hist_df = get_stock_hist_data(selected_stock)
            if hist_df is not None:
                fig = plot_stock_chart(hist_df, stock_info['名称'], selected_stock)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        # 风险提示
        st.markdown("---")
        st.error("⚠️ 风险提示: 本系统仅提供参考，不构成投资建议。务必设置止损单(建议-5%)，单只股票仓位不超过总资金的20%。")

    else:
        # 初始界面
        st.info("👈 设置好参数后，点击左侧「开始扫描市场」按钮进行扫描")

        st.markdown("---")
        st.subheader("📋 系统说明")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **筛选条件:**
            - MA20趋势筛选（股价在20日均线上方）
            - KDJ择时信号（金叉、超卖反弹）
            - 量价配合（量比 > 1.3）
            - 涨幅适中（避免追高）
            """)
        with col2:
            st.markdown("""
            **评分体系:**
            - 技术面评分 (60%权重)
            - 动量评分 (40%权重)
            - 优先级1: 上升趋势+KDJ共振
            - 优先级2: 趋势向上+KDJ超卖
            """)

if __name__ == "__main__":
    main()
