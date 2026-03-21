import streamlit as st
from sentiment_agent import fetch_batch_news, extract_quant_features
import yfinance as yf
import datetime

st.set_page_config(page_title="Quant Alpha System", layout="wide")
st.title("📊 机构级多模态 Alpha 挖掘系统")

col1, col2 = st.columns([1, 3])

with col1:
    ticker = st.text_input("输入交易标的 (如 NVDA):", "NVDA").upper()
    analyze_btn = st.button("生成研报与因子", use_container_width=True)

with col2:
    if analyze_btn:
        with st.spinner("系统正在进行高频抓取与多模态因子提取..."):
            # 1. 获取最新收盘价
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            latest_price = hist['Close'].iloc[-1] if not hist.empty else 0.0

            # 2. 获取新闻并提取因子
            raw_news = fetch_batch_news(ticker)
            factors = extract_quant_features(raw_news)

            # 3. 核心指标盘
            st.subheader(f"{ticker} 实时核心因子")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("最新收盘价", f"${latest_price:.2f}")
            c2.metric("综合情绪因子", factors['sentiment'])
            c3.metric("短期博弈因子", factors['short_term'])
            c4.metric("长期基本面因子", factors['long_term'])

            st.info(f"**Agent 交易逻辑判定**：{factors['reason']}")
