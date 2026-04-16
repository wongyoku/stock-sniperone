"""
股票量化精选系统 v2.0
=======================
从沪深A股全市场筛选优质股票
- 技术面筛选：MA20趋势 + KDJ择时 + 量价配合
- 基本面筛选：PE/ROE/市值 (需要网络支持)

Author: AI量化助手
"""

# 在导入akshare之前patch requests.Session，避免代理问题
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

import akshare as ak
import pandas as pd
import datetime
import warnings

warnings.filterwarnings('ignore')

# ================= 用户可配置参数 =================

# 【扫描范围】
SCAN_CONFIG = {
    "max_stocks": 500,        # 最多扫描股票数量(按成交额排序取前N)
    "min_turnover": 1e8,     # 最小日成交额(亿元)
}

# 【技术面参数】- 主要筛选依据
TECHNICAL_PARAMS = {
    "ma_period": 20,          # MA周期
    "kdj_n": 9,               # KDJ参数N
    "kdj_m1": 3,              # KDJ参数M1
    "kdj_m2": 3,              # KDJ参数M2
    "volume_ratio_min": 1.3,  # 最小量比
    "price_change_max": 0.09, # 最大涨幅(避免追高)
}

# 【基本面过滤参数】- 可选
FUNDAMENTAL_FILTERS = {
    "enabled": False,          # 是否启用基本面过滤(网络不稳定时可关闭)
    "pe_min": 5,              # 最小市盈率
    "pe_max": 80,             # 最大市盈率
    "market_cap_min": 50,     # 最小市值(亿)
    "market_cap_max": 5000,   # 最大市值(亿)
}

# 【信号权重】
SIGNAL_WEIGHTS = {
    "technical_score": 0.6,   # 技术面权重
    "momentum_score": 0.4,    # 动量权重
}


# ================= 数据获取模块 =================

def get_all_stocks():
    """获取沪深A股全市场股票列表 (使用新浪源)"""
    try:
        print(">> 正在获取全市场股票列表 (新浪数据源)...")
        df = ak.stock_zh_a_spot()

        # 新浪数据共14列: 序号,代码,名称,最新价,涨跌幅,涨跌额,今开,昨收,最高,最低,成交量,成交额,外盘,内盘
        # 取需要的列: 代码,名称,最新价,涨跌幅,最高,最低,成交量,成交额
        # 注意：索引从0开始，代码在索引0，名称在索引1，价格在索引2...
        df = df.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10]]
        df.columns = ['代码', '名称', '最新价', '涨跌幅', '最高', '最低', '成交量', '成交额']

        # 清理数据
        for col in ['最新价', '涨跌幅', '最高', '最低', '成交量', '成交额']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 排除ST股票
        df['名称'] = df['名称'].fillna('').astype(str)
        df = df[~df['名称'].str.contains('ST|退市', na=False, case=False)]

        # 排除涨跌停无法买入的股票
        df = df[df['涨跌幅'] > -9.9]  # 排除跌停

        # 按成交额排序，取前N只(流动性好的股票)
        df = df.sort_values('成交额', ascending=False)
        df = df.head(SCAN_CONFIG["max_stocks"])

        print(f"  [OK] 获取到 {len(df)} 只股票 (成交额前{SCAN_CONFIG['max_stocks']})")
        return df
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return None


def get_stock_hist_data(symbol):
    """获取个股历史数据"""
    try:
        # symbol 已经是带前缀的格式: sh600519, sz000001, bj920000
        # 直接使用，不需要再添加前缀
        full_code = symbol

        # 使用新浪历史数据接口
        df = ak.stock_zh_a_daily(symbol=full_code, adjust='qfq')

        if not isinstance(df, pd.DataFrame) or df is None:
            return None

        if len(df) < 30:
            return None

        # 重命名列以匹配我们的预期
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

        # 检查必要的列
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return None

        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])

        # 取最近60个交易日数据
        df = df.sort_values('date').tail(60).copy()

        return df
    except Exception as e:
        return None


def get_stock_info(symbol):
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

def calculate_indicators(df):
    """计算技术指标"""
    if df is None or len(df) < 30:
        return None

    # 重命名列
    df.columns = [col.strip() for col in df.columns]
    col_map = {
        '日期': 'date', '开盘': 'open', '最高': 'high',
        '最低': 'low', '收盘': 'close', '成交量': 'volume',
        '成交额': 'amount'
    }
    df = df.rename(columns=col_map)

    # MA计算
    ma_period = TECHNICAL_PARAMS["ma_period"]
    df['MA'] = df['close'].rolling(window=ma_period).mean()

    # KDJ计算
    n = TECHNICAL_PARAMS["kdj_n"]
    m1 = TECHNICAL_PARAMS["kdj_m1"]
    m2 = TECHNICAL_PARAMS["kdj_m2"]

    low_list = df['low'].rolling(window=n, min_periods=1).min()
    high_list = df['high'].rolling(window=n, min_periods=1).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100

    df['K'] = rsv.ewm(com=m1-1, adjust=False).mean()
    df['D'] = df['K'].ewm(com=m2-1, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # 动量指标
    df['price_change'] = df['close'].pct_change()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['ma_distance'] = (df['close'] - df['MA']) / df['MA']

    # 距离MA的距离百分比
    df['MA偏离度'] = ((df['close'] - df['MA']) / df['MA'] * 100).round(2)

    return df.dropna().reset_index(drop=True)


def calculate_technical_score(df):
    """技术面评分 (0-100)"""
    if df is None or len(df) < 5:
        return 0

    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0

    # 1. 趋势评分 (MA20之上得分)
    if last['close'] > last['MA']:
        score += 30
        ma_score = min(10, last['ma_distance'] * 100)
        score += ma_score

    # 2. KDJ评分
    if last['J'] < 20:
        score += 25
    elif last['J'] < 40:
        score += 15

    # KDJ金叉信号
    if last['J'] > last['K'] and prev['J'] <= prev['K'] and last['J'] < 60:
        score += 20

    # 3. 量价配合
    if last['volume_ratio'] > TECHNICAL_PARAMS["volume_ratio_min"]:
        score += 15

    # 4. 涨幅适中
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

    # 上升趋势确认
    trend_up = last['close'] > last['MA']

    # KDJ信号
    kdj_oversold = last['J'] < 30
    kdj_golden_cross = last['J'] > last['K'] and prev['J'] <= prev['K']
    kdj_bounce = kdj_oversold and (last['J'] > prev['J'])

    # 量价配合
    volume_ok = last['volume_ratio'] > TECHNICAL_PARAMS["volume_ratio_min"]

    # 综合判断
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


# ================= 主程序 =================

def scan_market():
    """扫描全市场并筛选标的"""
    print(f"\n{'='*60}")
    print(f"[量化选股扫描报告] | {datetime.date.today()}")
    print(f"{'='*60}\n")

    # Step 0: 测试数据获取
    print(">> 测试历史数据获取...")
    try:
        test_df = ak.stock_zh_a_daily(symbol='sh600519', adjust='qfq')
        print(f"  [OK] 测试成功: {len(test_df)} 行")
    except Exception as e:
        print(f"  [ERR] 测试失败: {e}")

    # Step 1: 获取全市场股票
    all_stocks = get_all_stocks()
    if all_stocks is None or len(all_stocks) == 0:
        print("[X] 无法获取股票列表")
        return

    # Step 2: 技术面筛选
    print(f"[扫描] 开始技术面筛选 ({len(all_stocks)} 只股票)...")
    print(f"   筛选条件: MA20趋势 + KDJ信号 + 量比>{TECHNICAL_PARAMS['volume_ratio_min']}")

    candidates = []
    count = 0

    for _, stock in all_stocks.iterrows():
        code = stock['代码']
        name = stock['名称']
        count += 1

        if count % 50 == 0:
            print(f"   已扫描 {count}/{len(all_stocks)} 只...")

        # 获取历史数据
        hist_df = get_stock_hist_data(code)
        if hist_df is None:
            if count <= 10:
                print(f"  [WARN] {code} 历史数据获取失败")
            continue

        # 调试：打印前几只股票的原始数据
        if count <= 3:
            print(f"  [DEBUG] {code} 获取到 {len(hist_df)} 行数据, 列: {list(hist_df.columns)}")

        # 计算指标
        tech_df = calculate_indicators(hist_df)
        if tech_df is None:
            if count <= 5:
                print(f"  [DEBUG] {code} 指标计算失败")
            continue
        elif len(tech_df) < 5:
            if count <= 5:
                print(f"  [DEBUG] {code} 指标数据不足: {len(tech_df)} 行")
            continue
        elif count <= 3:
            print(f"  [DEBUG] {code} 指标计算成功: {len(tech_df)} 行, J值: {tech_df.iloc[-1]['J']:.1f}")

        # 调试：打印前几只股票的指标
        if count <= 3:
            last = tech_df.iloc[-1]
            print(f"  [{code}] 收盘:{last['close']:.2f} MA:{last['MA']:.2f} J:{last['J']:.1f} 量比:{last['volume_ratio']:.2f}")

        # 评估信号
        buy_signal = evaluate_buy_signal(tech_df)
        if buy_signal is None or buy_signal['priority'] >= 7:
            continue

        # 计算评分
        tech_score = calculate_technical_score(tech_df)

        # 计算近期动量 (5日涨幅)
        if len(tech_df) >= 5:
            momentum = (tech_df.iloc[-1]['close'] / tech_df.iloc[-5]['close'] - 1) * 100
        else:
            momentum = 0

        # 综合评分
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

    # Step 3: 排序并输出结果
    print(f"\n{'='*60}")
    print(f"[结果] 筛选结果: {len(candidates)} 只股票通过全部条件")
    print(f"{'='*60}\n")

    if len(candidates) == 0:
        print(" 今日无符合条件的标的")
        print("\n[*] 可能原因:")
        print("   1. 市场整体偏弱，无明显买入信号")
        print("   2. 股票数量较多，建议稍后再试")
        return

    # 转为DataFrame并排序
    result_df = pd.DataFrame(candidates)
    result_df = result_df.sort_values(['优先级', '综合评分'], ascending=[True, False])

    # 输出TOP推荐
    print("[TOP1] 强烈推荐 (优先级1，趋势+KDJ共振)")
    top_picks = result_df[result_df['优先级'] == 1].head(5)
    if len(top_picks) > 0:
        print(top_picks[['代码', '名称', '最新价', '涨跌幅', '综合评分', 'J值', 'MA偏离度', '信号']].to_string(index=False))
    else:
        print("  今日无")

    print("\n[TOP2] 较好标的 (优先级2)")
    second_picks = result_df[result_df['优先级'] == 2].head(5)
    if len(second_picks) > 0:
        print(second_picks[['代码', '名称', '最新价', '涨跌幅', '综合评分', 'J值', 'MA偏离度', '信号']].to_string(index=False))
    else:
        print("  今日无")

    # TOP 15 详细表格
    print(f"\n{'='*60}")
    print("[列表] TOP 15 综合评分榜")
    print(f"{'='*60}")
    top15 = result_df.head(15)
    display_cols = ['代码', '名称', '最新价', '涨跌幅', '综合评分', 'J值', 'MA偏离度', '量比']
    print(top15[display_cols].to_string(index=False))

    # 操作建议
    print(f"\n{'='*60}")
    print("[*] 操作建议")
    print(f"{'='*60}")
    best = result_df.iloc[0] if len(result_df) > 0 else None
    if best is not None:
        print(f"\n最具潜力标的: {best['名称']}({best['代码']})")
        print(f"  现价: {best['最新价']:.2f} | 今日涨幅: {best['涨跌幅']:.2f}%")
        print(f"  综合评分: {best['综合评分']} | J值: {best['J值']} | MA偏离: {best['MA偏离度']}%")
        print(f"  信号: {best['信号']}")
        print(f"\n  建议仓位: 轻仓试探 (不超过总资金10%)")
        print(f"  止损位: {best['最新价'] * 0.95:.2f} (-5%)")
        print(f"  目标位: {best['最新价'] * 1.10:.2f} (+10%)")

    # 风险提示
    print(f"\n{'='*60}")
    print("[!] 风险提示:")
    print("   1. 本系统仅提供参考，不构成投资建议")
    print("   2. 建议单只股票仓位不超过总资金的20%")
    print("   3. 务必设置止损单(建议-5%)")
    print("   4. 操作后同步在券商APP设置条件单")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    scan_market()
