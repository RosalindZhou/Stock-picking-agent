# OracleAgent 📊 机构级多模态 Alpha 挖掘系统

## 📝 项目简介
OracleAgent 是一个驱动于大模型与机器学习的量化交易辅助预测系统。它通过融合市场新闻的非结构化数据（情绪面）与股票的真实量价数据（技术面），构建多模态特征集，进而训练 LightGBM 模型以预测标的未来的短期收益率。系统配备了极具科技感的交互式前端面板，实现从数据抓取、因子推演到可视化看板的全链路闭环。

## ✨ 核心特性

* **多模态因子提取**：系统内置 `sentiment_agent.py`，使用 `yfinance` 抓取实时新闻，并调用 Google Gemini 大模型（配置 Pydantic Schema）强制输出结构化的综合情绪、短期冲击和长期基本面因子。
* **严谨的机器学习预测**：机器学习引擎融合了 AI 情绪因子与真实的量价技术指标（5日波动率、异常流动性 Turnover_Proxy、RSI_14），以预测 T+3 日的真实收益率。
* **防数据穿越机制**：在模型训练阶段严格采用 `TimeSeriesSplit` 进行时间序列滚动窗口交叉验证，杜绝未来函数，确保预测模型的真实有效性。
* **极客风可视化数据看板**：提供支持中美双市场（美股/A股）切换的前端界面，内置红绿灯交易逻辑判定、动态 K 线与目标价位图表（基于 ECharts）以及动态置信度进度条。

## 📂 核心文件结构

* **`frontend/index.html`**: 系统的可视化前端界面，采用毛玻璃（Glassmorphism）UI 设计，负责渲染预测图表、展示因子评估数据并与后端 API 交互。
* **`sentiment_agent.py`**: NLP 情绪分析代理，负责从雅虎财经拉取新闻，并让大模型生成包含各类情绪得分（-1.0 到 1.0）及核心逻辑的分析结果。
* **`train_real.py`**: 真实的量化模型训练引擎，负责读取历史情绪因子文件，合成技术面指标，并训练 LightGBM 树模型，最终将最优权重保存为 `lgb_quant_model.txt` 供系统调用。

## 🚀 快速开始

### 1. 环境准备
确保你已安装 Python 3.8+ 环境。安装必要的依赖：
```bash
pip install yfinance pandas numpy lightgbm scikit-learn google-genai pydantic
