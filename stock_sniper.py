"""
股票量化精选系统 v2.0
=======================
从沪深A股全市场筛选优质股票
- 技术面筛选：MA20趋势 + KDJ择时 + 量价配合
- 基本面筛选：PE/ROE/市值 (需要网络支持)

Author: AI量化助手
"""

from stock_sniper_core import (
    ScanConfig, TechnicalParams, SignalWeights,
    get_all_stocks, get_stock_hist_data, get_stock_info,
    calculate_indicators, calculate_technical_score, evaluate_buy_signal,
    calculate_momentum, calculate_composite_score,
    scan_market_progress,
)
import akshare as ak
import pandas as pd
import datetime


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
    print(f"   筛选条件: MA20趋势 + KDJ信号 + 量比>{TechnicalParams.volume_ratio_min}")

    count = [0]  # mutable container for closure

    def progress_callback(idx, total, code, name):
        count[0] += 1
        if count[0] % 50 == 0:
            print(f"   已扫描 {count[0]}/{total} 只...")

    candidates_df = scan_market_progress(all_stocks, progress_callback)

    # Step 3: 排序并输出结果
    print(f"\n{'='*60}")
    print(f"[结果] 筛选结果: {len(candidates_df) if candidates_df is not None else 0} 只股票通过全部条件")
    print(f"{'='*60}\n")

    if candidates_df is None or len(candidates_df) == 0:
        print("  今日无符合条件的标的")
        print("\n[*] 可能原因:")
        print("   1. 市场整体偏弱，无明显买入信号")
        print("   2. 股票数量较多，建议稍后再试")
        return

    candidates_df = candidates_df.sort_values(['优先级', '综合评分'], ascending=[True, False])

    # 输出TOP推荐
    print("[TOP1] 强烈推荐 (优先级1，趋势+KDJ共振)")
    top_picks = candidates_df[candidates_df['优先级'] == 1].head(5)
    if len(top_picks) > 0:
        print(top_picks[['代码', '名称', '最新价', '涨跌幅', '综合评分', 'J值', 'MA偏离度', '信号']].to_string(index=False))
    else:
        print("  今日无")

    print("\n[TOP2] 较好标的 (优先级2)")
    second_picks = candidates_df[candidates_df['优先级'] == 2].head(5)
    if len(second_picks) > 0:
        print(second_picks[['代码', '名称', '最新价', '涨跌幅', '综合评分', 'J值', 'MA偏离度', '信号']].to_string(index=False))
    else:
        print("  今日无")

    # TOP 15 详细表格
    print(f"\n{'='*60}")
    print("[列表] TOP 15 综合评分榜")
    print(f"{'='*60}")
    top15 = candidates_df.head(15)
    display_cols = ['代码', '名称', '最新价', '涨跌幅', '综合评分', 'J值', 'MA偏离度', '量比']
    print(top15[display_cols].to_string(index=False))

    # 操作建议
    print(f"\n{'='*60}")
    print("[*] 操作建议")
    print(f"{'='*60}")
    best = candidates_df.iloc[0] if len(candidates_df) > 0 else None
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
