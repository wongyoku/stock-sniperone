import akshare as ak
import pandas as pd
import datetime
import warnings

# 忽略 pandas 的一些格式警告，保持终端输出清爽
warnings.filterwarnings('ignore')

# ================= 核心配置区 =================

# 1. 我们的“高弹性猎物池”
TARGET_ETFS = {
    "588000": "科创50ETF",
    "512480": "半导体ETF",
    "159819": "人工智能ETF",
    "512000": "券商ETF"
}

# 2. 你的当前持仓记录（盘后手动更新）
# 如果你买入了，就在这里填入数据。如果空仓，保持大括号为空 {} 即可。
# 格式： "ETF代码": {"cost": 你的买入均价, "high": 买入后的最高价, "days": 持仓天数}
MY_POSITIONS = {
    # 示例：假设你昨天买入了科创50
    # "588000": {"cost": 0.820, "high": 0.835, "days": 1} 
}


# ================= 数据与指标模块 =================

def get_etf_data(symbol):
    """抓取 ETF 数据并进行严格的类型清洗"""
    try:
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date="20230101", adjust="qfq")
        df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
        for col in ['开盘', '最高', '最低', '收盘', '成交量']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"[{symbol}] 数据抓取失败: {e}")
        return None

def calculate_indicators(df, n=9, m1=3, m2=3):
    """计算核心指标：MA20 与 KDJ"""
    if df is None or len(df) < 20:
        return None
        
    # 计算 MA20
    df['MA20'] = df['收盘'].rolling(window=20).mean()
    
    # 计算 KDJ
    low_list = df['最低'].rolling(window=n, min_periods=1).min()
    high_list = df['最高'].rolling(window=n, min_periods=1).max()
    rsv = (df['收盘'] - low_list) / (high_list - low_list) * 100
    
    df['K'] = rsv.ewm(com=m1-1, adjust=False).mean()
    df['D'] = df['K'].ewm(com=m2-1, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # 清除前期无法计算出 MA20 的空数据
    return df.dropna().reset_index(drop=True)


# ================= 策略大脑模块 =================

def evaluate_buy_signal(df, name):
    """进攻端：MA20趋势过滤 + KDJ择时"""
    last_day = df.iloc[-1]
    prev_day = df.iloc[-2]
    
    is_trend_up = last_day['收盘'] > last_day['MA20']
    is_kdj_buy = last_day['J'] < 20 and last_day['J'] > prev_day['J']
    
    print(f"  > 现价: {last_day['收盘']:.3f} | MA20: {last_day['MA20']:.3f} | J值: {last_day['J']:.2f}")
    
    if is_trend_up and is_kdj_buy:
        return "【★★★ 强烈买入信号】处于上升趋势且短期极度超卖拐点，建议明日早盘建仓！"
    elif is_kdj_buy and not is_trend_up:
        return "【⚠️ 放弃买入】虽然KDJ超卖拐头，但跌破MA20，趋势走坏，放弃抄底。"
    elif last_day['J'] > 80:
        return "【💡 风险提示】J值进入超买区(>80)，无仓位切勿追高。"
    else:
        return "【观望】未进入伏击圈，耐心等待。"

def evaluate_sell_signal(df, name, position_info):
    """防守端：10%止盈 + 三级止损"""
    current_price = df.iloc[-1]['收盘']
    buy_price = position_info['cost']
    highest_price = position_info['high']
    holding_days = position_info['days']
    
    profit_pct = (current_price - buy_price) / buy_price
    print(f"  > 成本: {buy_price:.3f} | 现价: {current_price:.3f} | 当前浮盈: {profit_pct*100:.2f}% | 持仓天数: {holding_days}")
    
    # 1. 终极止盈
    if profit_pct >= 0.10:
        return "【💰 止盈卖出】已达成 10% 盈利目标，立刻落袋为安，本月 KPI 完成！"
        
    # 2. 第一级：硬止损 (5%)
    if profit_pct <= -0.05:
        return "【🩸 止损卖出】触发硬止损：亏损达 5%，切断亏损，坚决离场！"
        
    # 3. 第二级：移动保本与吊灯止损
    if highest_price >= buy_price * 1.05 and (highest_price - current_price) / highest_price >= 0.02:
         return "【🛡️ 保护卖出】触发吊灯止损：最高点回撤达 2%，保住利润锁定胜局！"
    elif highest_price >= buy_price * 1.03 and current_price <= buy_price:
         return "【🛡️ 保本卖出】触发移动保本：盈利曾超 3%，现跌回成本，平仓保本出局！"
         
    # 4. 第三级：时间止损
    if holding_days >= 3 and abs(profit_pct) < 0.01:
        return "【⏱️ 时间卖出】触发时间止损：持仓 3 天未脱离成本区，平仓换股释放资金！"
        
    return "【🔒 安全持仓】各项指标正常，继续持有，严格挂好条件单。"


# ================= 主程序执行 =================

def main():
    print(f"==================================================")
    print(f" 📊 首席理财师的量化扫描报告 | 日期: {datetime.date.today()}")
    print(f"==================================================\n")
    
    for symbol, name in TARGET_ETFS.items():
        print(f"🔎 正在扫描: {name} ({symbol})")
        df = get_etf_data(symbol)
        df_ind = calculate_indicators(df)
        
        if df_ind is None:
            continue
            
        # 检查是否在持仓中
        if symbol in MY_POSITIONS:
            print("  [状态: 当前持仓中，进行卖出逻辑体检]")
            signal = evaluate_sell_signal(df_ind, name, MY_POSITIONS[symbol])
            print(f"  {signal}\n")
        else:
            print("  [状态: 空仓观察中，进行买入逻辑寻觅]")
            signal = evaluate_buy_signal(df_ind, name)
            print(f"  {signal}\n")
            
    print("==================================================")
    print(" 💡 理财师叮嘱：\n 如果产生操作，请务必在东方财富App同步设置好【云条件单】。")
    print("==================================================")

if __name__ == "__main__":
    main()