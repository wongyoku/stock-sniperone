# 📈 股票量化精选系统

基于技术分析的 A 股量化选股工具，支持 KDJ、MACD、RSI、布林带等多种指标筛选。

## 功能特点

- 🔍 全市场扫描 - 快速筛选符合条件的股票
- 📊 技术分析 - K线图 + MA + KDJ + MACD + RSI + WR
- 🎯 智能评分 - 综合技术面评分排序
- 🔄 股票对比 - 多股票雷达图对比
- 💾 数据缓存 - 加快重复访问速度

## 筛选指标

| 指标 | 说明 |
|------|------|
| MA20 | 20日均线趋势 |
| KDJ | 随机指标，金叉/超卖信号 |
| 量比 | 成交量放大程度 |
| MACD | 趋势判断 |
| RSI | 多空力量对比 |
| WR | 威廉指标 |

## 快速部署

### 使用 Streamlit Cloud（一键部署）

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

1. 点击上方按钮或访问 [share.streamlit.io](https://share.streamlit.io)
2. 使用 GitHub 登录
3. 导入本仓库
4. 等待自动部署完成

### 本地运行

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/stock-sniper.git
cd stock-sniper

# 安装依赖
pip install -r requirements.txt

# 运行
streamlit run streamlit_app.py
```

## 项目结构

```
stock-sniper/
├── streamlit_app.py       # 主程序
├── stock_sniper_core.py    # 核心模块
├── requirements.txt        # 依赖列表
└── README.md              # 说明文档
```

## 风险提示

⚠️ 本系统仅供学习研究参考，不构成任何投资建议！股票投资有风险，入市需谨慎。

## License

MIT License