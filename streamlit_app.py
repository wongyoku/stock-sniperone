"""
股票量化精选系统 - Streamlit 可视化版
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
import pandas as pd
import datetime
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

from stock_sniper_core import (
    ScanConfig, TechnicalParams, SignalWeights,
    get_all_stocks, get_stock_hist_data,
    calculate_indicators, calculate_technical_score, evaluate_buy_signal,
    calculate_momentum,
    scan_market_progress,
)


# ================= 配置参数同步 =================

def sync_params():
    """同步侧边栏参数到core模块"""
    ScanConfig.max_stocks = st.session_state.get('max_stocks', 500)
    TechnicalParams.ma_period = st.session_state.get('ma_period', 20)
    TechnicalParams.kdj_n = st.session_state.get('kdj_n', 9)
    TechnicalParams.volume_ratio_min = st.session_state.get('volume_ratio_min', 1.3)


# ================= K线图绘制 =================

def plot_stock_chart(df, stock_name, stock_code):
    """绘制个股K线图 - 所有指标整合到一个图表，可切换显示"""
    if df is None or len(df) < 20:
        return None

    # 计算额外MA线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()

    # K线颜色
    k_colors = ['#ef5350' if df['close'].iloc[i] >= df['open'].iloc[i] else '#26a69a' for i in range(len(df))]
    macd_colors = ['#ef5350' if h >= 0 else '#26a69a' for h in df['MACD_hist']]

    # WR计算
    wr = (df['high'].rolling(14).max() - df['close']) / (df['high'].rolling(14).max() - df['low'].rolling(14).min()) * -100

    fig = go.Figure()

    # 0: K线
    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='K线', increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
        increasing_fillcolor='#ef5350', decreasing_fillcolor='#26a69a', visible=True,
    ))

    # 1: MA5
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], name='MA5',
        line=dict(color='#ff6b6b', width=1.5), visible=True))

    # 2: MA10
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA10'], name='MA10',
        line=dict(color='#ffd93d', width=1.5), visible=True))

    # 3: MA20
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], name='MA20',
        line=dict(color='#6bcb77', width=2), visible=True))

    # 4: 布林带上轨
    fig.add_trace(go.Scatter(x=df['date'], y=df['BB_upper'], name='BB上轨',
        line=dict(color='rgba(150,150,150,0.6)', width=1), visible=True))

    # 5: 布林带下轨
    fig.add_trace(go.Scatter(x=df['date'], y=df['BB_lower'], name='BB下轨',
        line=dict(color='rgba(150,150,150,0.6)', width=1), visible=True))

    # 6: 布林带中轨
    fig.add_trace(go.Scatter(x=df['date'], y=df['BB_middle'], name='BB中轨',
        line=dict(color='rgba(150,150,150,0.6)', width=1, dash='dot'), visible=True))

    # 7: 成交量
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='成交量',
        marker_color=k_colors, opacity=0.6, visible=True, yaxis='y2'))

    # 8: KDJ-K
    fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name='K',
        line=dict(color='#2196f3', width=1.5), visible=False, yaxis='y3'))

    # 9: KDJ-D
    fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name='D',
        line=dict(color='#9c27b0', width=1.5), visible=False, yaxis='y3'))

    # 10: KDJ-J
    fig.add_trace(go.Scatter(x=df['date'], y=df['J'], name='J',
        line=dict(color='#ff9800', width=1.5), visible=False, yaxis='y3'))

    # 11: MACD柱
    fig.add_trace(go.Bar(x=df['date'], y=df['MACD_hist'], name='MACD柱',
        marker_color=macd_colors, opacity=0.8, visible=False, yaxis='y4'))

    # 12: DIF
    fig.add_trace(go.Scatter(x=df['date'], y=df['MACD'], name='DIF',
        line=dict(color='#2196f3', width=1.5), visible=False, yaxis='y4'))

    # 13: DEA
    fig.add_trace(go.Scatter(x=df['date'], y=df['MACD_signal'], name='DEA',
        line=dict(color='#9c27b0', width=1.5), visible=False, yaxis='y4'))

    # 14: RSI
    fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name='RSI',
        line=dict(color='#2196f3', width=2), visible=False, yaxis='y5'))

    # 15: WR
    fig.add_trace(go.Scatter(x=df['date'], y=wr, name='WR',
        line=dict(color='#e91e63', width=1.5), visible=False, yaxis='y6'))

    # visible数组索引: [K线, MA5, MA10, MA20, BB上, BB下, BB中, 成交量, K, D, J, MACD柱, DIF, DEA, RSI, WR]
    # 共16个traces (0-15)

    fig.update_layout(
        title=dict(text=f'{stock_name}({stock_code})', font=dict(size=16)),
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=60, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',

        yaxis=dict(title="", side="right", showgrid=True, gridcolor='rgba(200,200,200,0.3)', domain=[0.5, 1]),
        yaxis2=dict(title="", side="left", showgrid=False, overlaying="y", domain=[0.25, 0.45]),
        yaxis3=dict(title="", side="right", showgrid=True, gridcolor='rgba(200,200,200,0.3)', overlaying="y", domain=[0.18, 0.28], showticklabels=True),
        yaxis4=dict(title="", side="right", showgrid=True, gridcolor='rgba(200,200,200,0.3)', overlaying="y", domain=[0.10, 0.20], showticklabels=True),
        yaxis5=dict(title="", side="right", showgrid=True, gridcolor='rgba(200,200,200,0.3)', overlaying="y", domain=[0.02, 0.12], showticklabels=True),
        yaxis6=dict(title="", side="right", showgrid=True, gridcolor='rgba(200,200,200,0.3)', overlaying="y", domain=[0, 0.10], showticklabels=True),

        updatemenus=[
            dict(
                type="dropdown", direction="down", showactive=True,
                x=0.0, xanchor="left", y=1.15, yanchor="top",
                buttons=[
                    dict(label="📊 全部显示", method="update",
                        args=[{"visible": [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]},
                             {"title": f'{stock_name}({stock_code}) - 全部指标'}]),
                    dict(label="📈 K线+均线", method="update",
                        args=[{"visible": [True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False]},
                             {"title": f'{stock_name}({stock_code}) - K线+均线+布林带'}]),
                    dict(label="📉 成交量", method="update",
                        args=[{"visible": [False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False]},
                             {"title": f'{stock_name}({stock_code}) - 成交量'}]),
                    dict(label="🎯 KDJ", method="update",
                        args=[{"visible": [False, False, False, False, False, False, False, False, True, True, True, False, False, False, False, False]},
                             {"title": f'{stock_name}({stock_code}) - KDJ指标'}]),
                    dict(label="📊 MACD", method="update",
                        args=[{"visible": [False, False, False, False, False, False, False, False, False, False, False, True, True, True, False, False]},
                             {"title": f'{stock_name}({stock_code}) - MACD指标'}]),
                    dict(label="📏 RSI", method="update",
                        args=[{"visible": [False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False]},
                             {"title": f'{stock_name}({stock_code}) - RSI强弱指标'}]),
                    dict(label="🌊 WR威廉", method="update",
                        args=[{"visible": [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True]},
                             {"title": f'{stock_name}({stock_code}) - WR威廉指标'}]),
                ],
            )
        ]
    )
    return fig


# ================= 可视化辅助函数 =================

def plot_score_distribution(results):
    """评分分布图"""
    if results is None or len(results) == 0:
        return None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("综合评分分布", "技术评分 vs 综合评分"),
        specs=[[{"type": "histogram"}, {"type": "scatter"}]]
    )

    fig.add_trace(go.Histogram(
        x=results['综合评分'], nbinsx=20, name='综合评分',
        marker_color='#2196f3',
        hovertemplate='评分: %{x}<br>数量: %{y}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=results['技术评分'], y=results['综合评分'],
        mode='markers',
        marker=dict(size=9, color=results['优先级'], colorscale='RdYlGn_r', opacity=0.7),
        text=results['名称'],
        hovertemplate='%{text}<br>技术:%{x}<br>综合:%{y}<extra></extra>'
    ), row=1, col=2)

    fig.update_layout(
        height=300, showlegend=False, template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig.update_xaxes(title_text="评分", row=1, col=1)
    fig.update_xaxes(title_text="技术评分", row=1, col=2)
    fig.update_yaxes(title_text="数量", row=1, col=1)
    fig.update_yaxes(title_text="综合评分", row=1, col=2)
    return fig


def plot_priority_pie(results):
    """优先级分布饼图"""
    if results is None or len(results) == 0:
        return None

    priority_counts = results['优先级'].value_counts().sort_index()
    labels_map = {1: '强买入', 2: '中买入', 3: '轻买入', 4: '关注', 6: '信号不明'}
    pie_labels = [labels_map.get(p, f'优先级{p}') for p in priority_counts.index]

    fig = go.Figure(data=[go.Pie(
        labels=pie_labels, values=priority_counts.values, hole=0.4,
        marker_colors=['#26a69a', '#42a5f5', '#ab47bc', '#ff9800', '#78909c']
    )])
    fig.update_layout(height=280, showlegend=True, template="plotly_white", margin=dict(l=40, r=40, t=40, b=40))
    return fig


def plot_radar_compare(results, selected_codes):
    """股票对比雷达图"""
    if not selected_codes or len(selected_codes) < 2:
        return None

    selected = results[results['代码'].isin(selected_codes)]
    if len(selected) < 2:
        return None

    categories = ['综合评分', 'J值', 'MA偏离度', '量比']
    max_vals = {k: max(selected[k].max(), 1) for k in categories}

    fig = go.Figure()
    for _, row in selected.iterrows():
        values = [min(100, row[k] / max_vals[k] * 100) for k in categories]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            name=f"{row['名称']}({row['代码']})",
            fill='toself', opacity=0.6,
        ))

    fig.update_layout(
        height=350,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True, template="plotly_white",
    )
    return fig


# ================= 扫描市场 =================

@st.cache_data(ttl=3600)
def get_all_stocks_cached():
    return get_all_stocks()


def scan_market_ui():
    """扫描市场（极限加速版 - 流水线并行）"""
    import threading
    from queue import Queue

    all_stocks = get_all_stocks_cached()
    if all_stocks is None or len(all_stocks) == 0:
        return None

    stocks_list = list(all_stocks.itertuples(index=False))
    total = len(stocks_list)

    progress_bar = st.progress(0)
    status_text = st.empty()

    # 使用流水线：线程获取数据，进程计算指标
    # 数据队列
    data_queue = Queue(maxsize=1000)
    results_queue = Queue()
    completed_count = [0]
    count_lock = threading.Lock()

    # 第一阶段：80线程并行获取所有历史数据
    status_text.text("第一阶段：获取历史数据...")

    def fetch_worker(code):
        try:
            hist_df = get_stock_hist_data(code)
            if hist_df is not None:
                data_queue.put((code, hist_df), timeout=1)
        except:
            pass
        with count_lock:
            completed_count[0] += 1
            if completed_count[0] % 100 == 0 or completed_count[0] == total:
                progress = completed_count[0] / total * 0.35
                progress_bar.progress(progress)
                status_text.text(f"获取数据 {completed_count[0]}/{total}")

    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = [executor.submit(fetch_worker, stock.代码) for stock in stocks_list]
        # 等待所有获取完成
        for f in futures:
            f.result()

    # 标记获取完成
    data_queue.put(None)
    progress_bar.progress(0.35)
    status_text.text(f"数据获取完成，共 {data_queue.qsize()} 只")

    # 建立代码到股票信息的映射
    stock_info = {stock.代码: (stock.名称, stock.最新价, stock.涨跌幅) for stock in stocks_list}

    # 第二阶段：多进程并行计算指标
    status_text.text("第二阶段：计算指标...")

    from concurrent.futures import ProcessPoolExecutor

    def process_tasks():
        """在主线程中处理计算任务"""
        candidates = []
        batch = []
        batch_size = 50

        while True:
            item = data_queue.get()
            if item is None:
                break

            code, hist_df = item
            batch.append((code, hist_df))

            if len(batch) >= batch_size or data_queue.empty():
                # 处理这一批
                for code, hist_df in batch:
                    if code not in stock_info:
                        continue

                    name, latest_price, change_pct = stock_info[code]

                    try:
                        tech_df = calculate_indicators(hist_df)
                        if tech_df is None or len(tech_df) < 5:
                            continue

                        buy_signal = evaluate_buy_signal(tech_df)
                        if buy_signal is None or buy_signal['priority'] >= 7:
                            continue

                        tech_score = calculate_technical_score(tech_df)
                        momentum = calculate_momentum(tech_df)
                        total_score = tech_score * 0.6 + min(100, momentum + 50) * 0.4

                        last = tech_df.iloc[-1]
                        candidates.append({
                            "代码": code, "名称": name,
                            "最新价": latest_price, "涨跌幅": change_pct,
                            "技术评分": round(tech_score, 1), "综合评分": round(total_score, 1),
                            "J值": round(float(last['J']), 1),
                            "MA偏离度": float(last['MA偏离度']),
                            "量比": round(float(last['volume_ratio']), 2),
                            "信号": buy_signal['signal'],
                            "优先级": buy_signal['priority'],
                        })
                    except:
                        continue

                batch = []
                progress_bar.progress(0.35 + (1 - data_queue.qsize() / max(total, 1)) * 0.65)
                status_text.text(f"计算指标 {len(candidates)} 只符合条件")

        return candidates

    candidates = process_tasks()

    progress_bar.progress(100)
    status_text.text("扫描完成！")

    return pd.DataFrame(candidates) if candidates else None


# ================= Streamlit 页面 =================

st.set_page_config(page_title="股票量化精选", layout="wide")

# 自定义CSS
st.markdown("""
<style>
[data-testid="stSelectbox"] label {
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# 侧边栏参数
with st.sidebar:
    st.markdown("### 参数设置")

    max_stocks = st.slider("扫描数量", 100, 1000, 500, 50, key='max_stocks')
    ma_period = st.slider("MA周期", 5, 60, 20, 5, key='ma_period')
    kdj_n = st.slider("KDJ N", 3, 21, 9, key='kdj_n')
    volume_ratio = st.slider("量比阈值", 1.0, 3.0, 1.3, 0.1, key='volume_ratio_min')

    st.divider()
    st.markdown("### 股价区间筛选")

    price_min = st.number_input("最低价", min_value=0.0, max_value=1000.0, value=0.0, step=1.0, key='price_min')
    price_max = st.number_input("最高价", min_value=0.0, max_value=10000.0, value=100.0, step=1.0, key='price_max')

    # 同步到core模块
    ScanConfig.max_stocks = max_stocks
    TechnicalParams.ma_period = ma_period
    TechnicalParams.kdj_n = kdj_n
    TechnicalParams.volume_ratio_min = volume_ratio

    st.divider()
    st.markdown("""
    **指标说明**
    - KDJ：随机指标
    - MACD：趋势指标
    - RSI：强弱指标
    - BB：布林带
    - WR：威廉指标
    """)

# 主页面
st.markdown("## 📈 股票量化精选系统")
st.caption(f"{datetime.date.today()} · 沪深A股扫描")

scan_button = st.button("🔍 开始扫描", type="primary", use_container_width=True)

if scan_button or 'results' in st.session_state:
    if scan_button:
        with st.spinner("正在扫描市场..."):
            results = scan_market_ui()
            st.session_state['results'] = results
            st.session_state['scan_time'] = datetime.datetime.now()
    else:
        results = st.session_state.get('results')

    if results is None or len(results) == 0:
        st.warning("今日无符合条件的标的")
    else:
        results = results.sort_values(['优先级', '综合评分'], ascending=[True, False])

        # 股价区间筛选
        price_min = st.session_state.get('price_min', 0.0)
        price_max = st.session_state.get('price_max', 10000.0)
        results = results[(results['最新价'] >= price_min) & (results['最新价'] <= price_max)]

        if len(results) == 0:
            st.warning(f"当前股价区间 ({price_min:.2f} - {price_max:.2f}) 内无符合条件的标的")
            st.stop()

        st.caption(f"当前显示: {price_min:.2f} - {price_max:.2f} 元区间内 {len(results)} 只股票")

        # 概览指标
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("扫描股票", f"{max_stocks} 只")
        with col2: st.metric("符合条件", f"{len(results)} 只")
        with col3: st.metric("强买入", f"{len(results[results['优先级']==1])} 只")
        with col4: st.metric("平均评分", f"{results['综合评分'].mean():.1f}")

        st.divider()

        # 可视化
        col_chart, col_pie = st.columns([2, 1])
        with col_chart:
            dist_fig = plot_score_distribution(results)
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)
        with col_pie:
            pie_fig = plot_priority_pie(results)
            if pie_fig:
                st.plotly_chart(pie_fig, use_container_width=True)

        st.divider()

        # TOP推荐
        tab1, tab2, tab3 = st.tabs(["⭐ 强烈买入", "📈 较好标的", "📊 TOP30总榜"])

        with tab1:
            top1 = results[results['优先级']==1]
            if len(top1) > 0:
                st.dataframe(top1[['代码','名称','最新价','涨跌幅','综合评分','J值','MA偏离度']], use_container_width=True, hide_index=True)
            else:
                st.info("暂无信号")

        with tab2:
            top2 = results[results['优先级']==2].head(10)
            if len(top2) > 0:
                st.dataframe(top2[['代码','名称','最新价','涨跌幅','综合评分','J值','MA偏离度']], use_container_width=True, hide_index=True)
            else:
                st.info("暂无信号")

        with tab3:
            st.dataframe(results.head(30)[['代码','名称','最新价','涨跌幅','综合评分','J值','MA偏离度','量比','信号']], use_container_width=True, hide_index=True)

        st.divider()

        # 股票对比
        st.markdown("### 🔄 股票对比")
        cols = st.columns([1, 3])
        with cols[0]:
            selected = st.multiselect("选择股票", results['代码'].tolist(),
                default=results['代码'].head(3).tolist() if len(results) >= 3 else [],
                format_func=lambda x: f"{x} {results[results['代码']==x]['名称'].values[0].replace('（', '').replace('）', '')}")
        with cols[1]:
            if len(selected) >= 2:
                radar = plot_radar_compare(results, selected)
                if radar:
                    st.plotly_chart(radar, use_container_width=True)
            else:
                st.info("选择至少2只股票进行对比")

        st.divider()

        # 个股详情
        st.markdown("### 🔍 个股详情")
        col_sel = st.columns([2, 1])
        with col_sel[0]:
            stock_options = [f"{row['代码']} {row['名称']}" for _, row in results.iterrows()]
            selected_stock = st.selectbox("选择股票", stock_options, label_visibility="collapsed")
            selected_code = selected_stock.split(" ")[0]

        if selected_stock:
            info = results[results['代码']==selected_code].iloc[0]
            cols = st.columns(5)
            with cols[0]: st.metric("最新价", f"¥{info['最新价']:.2f}")
            with cols[1]: st.metric("涨跌幅", f"{info['涨跌幅']:.2f}%", delta=info['涨跌幅'])
            with cols[2]: st.metric("综合评分", f"{info['综合评分']}")
            with cols[3]: st.metric("J值", f"{info['J值']}")
            with cols[4]: st.metric("MA偏离度", f"{info['MA偏离度']}%")

            signal_color = "green" if info['优先级'] <= 2 else "orange"
            st.markdown(f"**信号:** :{signal_color}[{info['信号']}]")

            hist_df = get_stock_hist_data(selected_code)
            if hist_df is not None:
                tech_df = calculate_indicators(hist_df)
                if tech_df is not None:
                    fig = plot_stock_chart(tech_df, info['名称'], selected_code)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.caption("⚠️ 风险提示：本系统仅供参考，不构成投资建议")

else:
    st.info("👈 点击上方「开始扫描」按钮进行市场扫描")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **筛选逻辑**
        - MA20趋势筛选
        - KDJ金叉/超卖信号
        - 量价配合
        """)
    with col2:
        st.markdown("""
        **K线图指标**
        - MA5/MA10/MA20均线
        - Bollinger Bands
        - KDJ/MACD/RSI/WR
        - 成交量
        """)