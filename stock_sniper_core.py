"""
股票量化精选系统 - 核心模块
===========================
所有技术分析、数据获取、评分计算的共享逻辑

Author: AI量化助手
"""

import requests
import os
from typing import Optional

import akshare as ak
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ================= 请求环境修复 =================

for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]

_original_session = requests.Session

def _patched_session(*args, **kwargs):
    session = _original_session(*args, **kwargs)
    session.trust_env = False
    return session

requests.Session = _patched_session


# ================= 配置参数 =================

class ScanConfig:
    """扫描配置"""
    max_stocks: int = 500
    min_turnover: float = 1e8

    @classmethod
    def update(cls, **kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)


class TechnicalParams:
    """技术指标参数"""
    ma_period: int = 20
    kdj_n: int = 9
    kdj_m1: int = 3
    kdj_m2: int = 3
    volume_ratio_min: float = 1.3
    price_change_max: float = 0.09

    @classmethod
    def update(cls, **kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)


class SignalWeights:
    """信号权重"""
    technical_score: float = 0.6
    momentum_score: float = 0.4

    @classmethod
    def update(cls, **kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)


# ================= 数据获取模块 =================

def get_all_stocks() -> Optional[pd.DataFrame]:
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
        df = df.head(ScanConfig.max_stocks)

        return df
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return None


def get_stock_hist_data(symbol: str, lookback: int = 120) -> Optional[pd.DataFrame]:
    """获取个股历史数据

    Args:
        symbol: 股票代码 (如 sh600519, sz000001)
        lookback: 最多返回多少个交易日的数据
    """
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
        df = df.sort_values('date').tail(lookback).copy()

        return df
    except Exception:
        return None


def get_stock_info(symbol: str) -> Optional[dict]:
    """获取个股详细信息 (PE/ROE等)"""
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        info = {}
        for _, row in df.iterrows():
            info[row['item']] = row['value']
        return info
    except Exception:
        return None


# ================= 技术分析模块 =================

def calculate_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """计算技术指标 (MA, KDJ, 成交量等)"""
    if df is None or len(df) < 30:
        return None

    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    col_map = {
        '日期': 'date', '开盘': 'open', '最高': 'high',
        '最低': 'low', '收盘': 'close', '成交量': 'volume',
        '成交额': 'amount'
    }
    df = df.rename(columns=col_map)

    # MA
    ma_period = TechnicalParams.ma_period
    df['MA'] = df['close'].rolling(window=ma_period).mean()

    # KDJ
    n = TechnicalParams.kdj_n
    m1 = TechnicalParams.kdj_m1
    m2 = TechnicalParams.kdj_m2

    low_list = df['low'].rolling(window=n, min_periods=1).min()
    high_list = df['high'].rolling(window=n, min_periods=1).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100

    df['K'] = rsv.ewm(com=m1-1, adjust=False).mean()
    df['D'] = df['K'].ewm(com=m2-1, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std

    # 动量与量价指标
    df['price_change'] = df['close'].pct_change()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['ma_distance'] = (df['close'] - df['MA']) / df['MA']
    df['MA偏离度'] = ((df['close'] - df['MA']) / df['MA'] * 100).round(2)

    return df.dropna().reset_index(drop=True)


def calculate_technical_score(df: pd.DataFrame) -> float:
    """技术面评分 (0-100)"""
    if df is None or len(df) < 5:
        return 0.0

    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0.0

    # 趋势评分
    if last['close'] > last['MA']:
        score += 30
        ma_score = min(10, last['ma_distance'] * 100)
        score += ma_score

    # KDJ评分
    if last['J'] < 20:
        score += 25
    elif last['J'] < 40:
        score += 15

    if last['J'] > last['K'] and prev['J'] <= prev['K'] and last['J'] < 60:
        score += 20

    # 量价配合
    if last['volume_ratio'] > TechnicalParams.volume_ratio_min:
        score += 15

    # 涨幅适中
    change = last['close'] / df.iloc[-2]['close'] - 1 if len(df) > 1 else 0
    if 0 < change < TechnicalParams.price_change_max:
        score += 10
    elif change >= TechnicalParams.price_change_max:
        score += 3

    return min(100.0, score)


def evaluate_buy_signal(df: pd.DataFrame) -> Optional[dict]:
    """评估买入信号"""
    if df is None or len(df) < 5:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    trend_up = last['close'] > last['MA']
    kdj_oversold = last['J'] < 30
    kdj_golden_cross = last['J'] > last['K'] and prev['J'] <= prev['K']
    kdj_bounce = kdj_oversold and (last['J'] > prev['J'])
    volume_ok = last['volume_ratio'] > TechnicalParams.volume_ratio_min

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
        "J值": round(float(last['J']), 1),
        "MA偏离度": float(last['MA偏离度']),
        "量比": round(float(last['volume_ratio']), 2),
        "涨幅": round(float((last['close'] / df.iloc[-2]['close'] - 1) * 100), 2),
    }


def calculate_momentum(df: pd.DataFrame, periods: int = 5) -> float:
    """计算N日动量 (涨幅百分比)"""
    if df is None or len(df) < periods:
        return 0.0
    return float((df.iloc[-1]['close'] / df.iloc[-periods]['close'] - 1) * 100)


def calculate_composite_score(df: pd.DataFrame) -> float:
    """计算综合评分"""
    tech_score = calculate_technical_score(df)
    momentum = calculate_momentum(df)
    total = tech_score * SignalWeights.technical_score + \
            min(100.0, momentum + 50) * SignalWeights.momentum_score
    return round(total, 1)


# ================= K线图绘制 =================

def plot_stock_chart(df: pd.DataFrame, stock_name: str, stock_code: str):
    """绘制个股K线图+MA+KDJ+MACD+RSI

    Returns:
        plotly.graph_objects.Figure 对象，或 None
    """
    if df is None or len(df) < 20:
        return None

    ma_period = TechnicalParams.ma_period

    # 创建子图: K线+MA, 成交量, KDJ, MACD, RSI
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.35, 0.15, 0.15, 0.15, 0.15],
        subplot_titles=(
            f'{stock_name}({stock_code})',
            '成交量',
            'KDJ指标',
            'MACD指标',
            'RSI指标'
        )
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

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['BB_upper'], name='BB上轨',
                   line=dict(color='rgba(100,100,100,0.3)', width=1),
                   hoverinfo='skip'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['BB_lower'], name='BB下轨',
                   line=dict(color='rgba(100,100,100,0.3)', width=1),
                   hoverinfo='skip'),
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
        go.Scatter(x=df['date'], y=df['K'], name='K', line=dict(color='#0074D9', width=1.2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['D'], name='D', line=dict(color='#B10DC9', width=1.2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['J'], name='J', line=dict(color='#FF851B', width=1.2)),
        row=3, col=1
    )
    fig.add_hline(y=80, line_dash="dash", line_color="#FF4136", row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="#3D9970", row=3, col=1)

    # MACD
    macd_colors = ['#FF4136' if h >= 0 else '#3D9970' for h in df['MACD_hist']]
    fig.add_trace(
        go.Bar(x=df['date'], y=df['MACD_hist'], name='MACD柱',
               marker_color=macd_colors),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['MACD'], name='DIF',
                   line=dict(color='#0074D9', width=1.2)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['MACD_signal'], name='DEA',
                   line=dict(color='#B10DC9', width=1.2)),
        row=4, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['RSI'], name='RSI',
                   line=dict(color='#0074D9', width=1.2)),
        row=5, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#FF4136", row=5, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#3D9970", row=5, col=1)

    fig.update_layout(
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    return fig


# ================= 市场扫描 =================

def scan_market_progress(all_stocks: pd.DataFrame, progress_callback=None):
    """扫描全市场并筛选标的 (支持进度回调)

    Args:
        all_stocks: 股票列表 DataFrame
        progress_callback: 回调函数，接收 (当前索引, 总数, 当前股票信息) 参数

    Returns:
        符合条件的股票 DataFrame
    """
    if all_stocks is None or len(all_stocks) == 0:
        return None

    candidates = []

    for idx, (_, stock) in enumerate(all_stocks.iterrows()):
        code = stock['代码']
        name = stock['名称']

        if progress_callback:
            progress_callback(idx, len(all_stocks), code, name)

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
        momentum = calculate_momentum(tech_df)
        total_score = tech_score * SignalWeights.technical_score + \
                      min(100.0, momentum + 50) * SignalWeights.momentum_score

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

    if not candidates:
        return None

    return pd.DataFrame(candidates)


# ================= 导入放在末尾避免循环依赖 =================

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except ImportError:
    pass
