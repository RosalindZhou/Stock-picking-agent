# OracleAgent: 多模态量化分析与预测引擎

OracleAgent 是一套机构级标准的高集成度全栈金融量化架构。系统巧妙融合了传统技术量价指标（如 RSI, 波动率, 异常流动性极差等）与前沿的大语言模型（Gemini NLP）全网情绪分析，利用集成树模型（LightGBM）执行严苛的时序验证，从而捕捉具有高置信度的短线（如 T+3）复合 Alpha 因子收益。

## 🌟 核心技术突破

- **强结构化大模型解析 (JSON Schema & Pydantic)**
  告别传统且脆弱的正则表达式。系统底层通过强类型校验直接打通 Google Gemini Flash Lite 官方接口，确保大模型无论阅读多么长篇大论的非结构化新闻，都能 **100% 稳定地返回规范化 JSON 字典因子**（精准锚定情绪得分、短期抛压、长线基本面等维度），消除空值与解析乱码。

- **广谱双市场降级路由 (A-Share & US Market)**
  智能判断输入标的。系统不仅能通过 Finnhub 主力接口监控华尔街龙头（如 NVDA, QQQ），亦整合了 **AkShare** 进行底层穿透，秒级抓取东方财富网的 A 股（如 600519，东方财富等）实时财经脉搏与历史除权 K 线。

- **零未来函数时序切分 (Zero Look-Ahead Bias)**
  实盘即演习。在模型的批量训练链路中，极其严格地实装了 `TimeSeriesSplit` 滚动窗口验证取代随机切分。确保“过去的特征”永远无法提前窥视“未来的涨跌”，高度还原充满迷雾的真实二级市场回撤盲测场景。

- **灰度风控熔断灾备**
  当外部数据源超时、大模型 API 频控耗尽、甚至出现断网情况时，系统的兜底风控网将自动截断脏数据透传，强制接管并向前台抛出“🟡 灰度风控”黄灯状态，规避自动量化因空数据而可能引发的灾难性交易指令。

- **金融终端级前瞻看板 (ECharts Native)**
  抛弃枯燥的纯数据表谱，前端构建了一套带 DataZoom 自由极速缩放的高性能 K 线沙盘。
  自动计算叠加 5 / 10 日智能动量均线 (MA5/10)，创新型地引入带全息阴影的 T+3 AI 预期标靶点 (`markPoint`)；同时原生适配跨文化配色痛点（A股红涨绿跌，美股绿涨红跌全兼容）。

## 🚀 快速本地启动

1. **环境准备**  
   需要 Python 3.9+，通过运行安装核心依赖：
   ```bash
   pip install fastapi uvicorn requests pandas numpy yfinance akshare google-genai pydantic lightgbm
   ```

2. **配置私有 API 鉴权库**  
   在根目录下新建一个 `.env` 隐藏文件，填入您的密钥（该文件已被 `.gitignore` 严厉隔离，确保 Github 开源脱敏）：
   ```env
   FINNHUB_API_KEY=YOUR_FINNHUB_KEY
   GOOGLE_API_KEY=YOUR_GEMINI_KEY
   ```

3. **激发服务端马达**  
   ```bash
   python api_server.py
   ```
   *服务将在 `http://localhost:8000` 监听侦测*

4. **开启量化沙盘**  
   双击 `frontend/index.html` 即可在本地浏览器中享受顺滑的高级推演大图。
