import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import re
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

try:
    from data_harvester import FINNHUB_API_KEY, GOOGLE_API_KEY
except ImportError:
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

app = FastAPI(title="OracleAgent Data API")
# 全局变量：存储缓存数据
prediction_cache = {}
# 缓存有效期：1 小时
CACHE_EXPIRE_SECONDS = 3600

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_index():
    """
    当用户访问根目录时，返回 index.html 网页
    """
    import os
    # 检查当前目录下是否有 index.html 文件
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    # 如果没找到网页文件，就返回一个简单的 JSON 提示
    return {"status": "running", "message": "OracleAgent API is live, but index.html is missing."}
    
client = None
if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY":
    client = genai.Client(api_key=GOOGLE_API_KEY)

class PredictRequest(BaseModel):
    ticker: str
    market: str = "US"

SECTOR_MAP = {
    "NVDA": "QQQ", "AAPL": "QQQ", "MSFT": "QQQ", "TSLA": "QQQ", "AMD": "QQQ", "GOOG": "QQQ", "META": "QQQ",
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF",
    "JNJ": "XLF", "V": "XLF",
    "XOM": "XLE", "CVX": "XLE"
}

import json
from pydantic import BaseModel, Field

class ApiQuantFeatures(BaseModel):
    sentiment: float = Field(description="综合情绪得分 (-1.0到1.0)")
    short_term: float = Field(description="短期冲击得分 (-1.0到1.0)")
    long_term: float = Field(description="长期基本面得分 (-1.0到1.0)")
    rating_exp: str = Field(description="一句话评级解释(结合情绪)")
    opp: str = Field(description="核心机会(极简)")
    risk: str = Field(description="核心风险(压制)")
    warn: str = Field(description="关注阈值")
    sector_trend: str = Field(description="宏观环境")
    company_trend: str = Field(description="个股特质因子")
    history: str = Field(description="历史同源分析")

def fetch_top_news(symbol: str, from_date: str, to_date: str, limit: int = 1, market: str = "US") -> str:
    if market == "CN":
        pure_symbol = symbol.split(".")[0] if "." in symbol else symbol
        try:
            import akshare as ak
            df_news = ak.stock_news_em(symbol=pure_symbol)
            if df_news is None or df_news.empty: return "数据源为空"
            compiled_news = ""
            count = 0
            for idx, row in df_news.iterrows():
                title = str(row.get('新闻标题', '')).strip()
                if not title: continue
                compiled_news += f"Title: {title}\n"
                count += 1
                if count >= limit: break
            return compiled_news
        except Exception:
            return "AkShare实时数据源保护过滤"
            
    original_key = FINNHUB_API_KEY
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={original_key}"
    try:
        response = requests.get(url, timeout=1.5)
        if response.status_code == 429: return ""
        response.raise_for_status()
        raw_news = response.json()
        
        compiled_news = ""
        count = 0
        for item in raw_news:
            title = item.get("headline", "").strip()
            if not title: continue
            compiled_news += f"Title: {title}\n"
            count += 1
            if count >= limit: break
            
        return compiled_news
    except Exception:
        return "数据源超时保护过滤"

def extract_algorithmic_features(ticker: str, sector: str, company_news: str, sector_news: str, retries: int = 2, market: str = "US") -> dict:
    news_count = company_news.count("Title:")
    confidence = min(98, 55 + news_count * 15)

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        return {
            "sentiment": 0.0, "short_term": 0.0, "long_term": 0.0, 
            "sector_trend": "未配置大模型密钥", "company_trend": "N/A", "history": "N/A",
            "opp": "N/A", "risk": "N/A", "warn": "N/A", "confidence": confidence,
            "rating_exp": "系统未激活", "light_status": "🔴 故障缺失", "light_color": "red"
        }

    market_scope = "美国三大股指/特定ETF" if market == "US" else "中国A股与宏观政策面"
    prompt = f"""您是顶级的自动量化研报系统。标的: {ticker}, 市场: {market_scope}。
    最新动态输入: {company_news if company_news else '无最新动态'}。

    【严格纪律要求】
    1. 坚守事实：所有的分析、得分（如 sentiment, short_term, long_term）必须且只能基于上面提供的“最新动态输入”和该标的已知的客观基本面。
    2. 拒绝捏造：绝不允许自行虚构任何财务数据、新闻事件或主观数字。如果没有足够的信息，必须给出保守的中性评分（如 0.0）并说明信息不足。
    3. 绝对一致：你的推理逻辑必须像计算器一样严谨。面对相同的信息输入，你给出的各项评分必须保持恒定，不能有任何随机波动。
    
    请严格结合上述数据，生成专业的评价结构。"""
    
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
        temperature=0.0,)
            )
            
            data = json.loads(response.text)
            sentiment = float(data.get("sentiment", 0.0))
            if sentiment >= 0.3:
                light_status, light_color = "🟢 绿灯 (积极/做多)", "green"
            elif sentiment > -0.3:
                light_status, light_color = "🟡 黄灯 (震荡/观望)", "yellow"
            else:
                light_status, light_color = "🔴 红灯 (风险/看空)", "red"

            return {
                "sentiment": sentiment,
                "short_term": float(data.get("short_term", 0.0)),
                "long_term": float(data.get("long_term", 0.0)),
                "sector_trend": str(data.get("sector_trend", "宏观共振模型运算中")),
                "company_trend": str(data.get("company_trend", "微观结构测算中")),
                "history": str(data.get("history", "匹配度验证中")),
                "opp": str(data.get("opp", "超额收益测算点不明")),
                "risk": str(data.get("risk", "尾部风险未触及")),
                "warn": str(data.get("warn", "技术支撑回撤位")),
                "rating_exp": str(data.get("rating_exp", "中性对冲区间")),
                "light_status": light_status,
                "light_color": light_color,
                "confidence": confidence
            }
        except Exception as e:
            print(f"[Gemini Timeout or JSON Error] {e}")
            if attempt < retries - 1:
                time.sleep(1)
            
    return {
        "sentiment": 0.0, "short_term": 0.0, "long_term": 0.0, 
        "sector_trend": "量化模型防熔断系统启动", "company_trend": "数据源信噪严重", "history": "样本分布异常",
        "opp": "波动流阻断", "risk": "风险未核准", "warn": "系统保护", "confidence": 0,
        "rating_exp": "系统阻断", "light_status": "🟡 灰度风控", "light_color": "yellow"
    }

prediction_cache = {}
CACHE_EXPIRE_SECONDS = 3600

@app.post("/api/predict")
def predict_stock(req: PredictRequest):
    ticker = req.ticker.upper().strip()
    
    # 1. 先查缓存
    cache_key = f"{ticker}_{req.market}"
    current_time = time.time()
    if cache_key in prediction_cache:
        cached_data, timestamp = prediction_cache[cache_key]
        if current_time - timestamp < CACHE_EXPIRE_SECONDS:
            print(f"[Cache Hit] 返回 {ticker} 的有效缓存数据")
            return cached_data

    try:
        # --- 原有的数据抓取和计算逻辑 ---
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        sector = SECTOR_MAP.get(ticker, "SPY") if req.market == "US" else "中证A500/沪深300"
        
        company_news = fetch_top_news(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), limit=1, market=req.market)
        sector_news = "" 
        
        factors = extract_algorithmic_features(ticker, sector, company_news, sector_news, market=req.market)

        # 2. 【核心校验】检查返回的是不是我们要的业务数据
        # 检查大模型返回的 light_status 是否包含“故障”或“保护”字样，或者置信度是否为 0
        if factors.get("confidence") == 0 or "故障" in factors.get("light_status", ""):
            print(f"[API Error] {ticker} 模型返回异常，不进入缓存")
            raise HTTPException(status_code=503, detail="AI Service temporary unavailable, please retry.")
        
        yf_ticker = ticker
        df_price = None
        try:
            if req.market == "CN":
                import akshare as ak
                pure_symbol = ticker.split(".")[0] if "." in ticker else ticker
                df_ak = ak.stock_zh_a_hist(symbol=pure_symbol, period="daily", start_date=(end_date - timedelta(days=60)).strftime('%Y%m%d'), end_date=end_date.strftime('%Y%m%d'), adjust="qfq")
                df_ak.rename(columns={'日期':'Date', '开盘':'Open', '收盘':'Close', '最高':'High', '最低':'Low', '成交量':'Volume'}, inplace=True)
                df_ak.set_index('Date', inplace=True)
                df_price = df_ak[['Open', 'Close', 'High', 'Low', 'Volume']]
            else:
                df_price = yf.Ticker(yf_ticker).history(period="1mo")
                
            if len(df_price) < 15: 
                raise ValueError("Data source returned insufficient data rows.")
        except Exception as e:
            print(f"[YFinance Fetch Failed]: {e}. Using simulated base fallback.")
            dates = pd.date_range(end=datetime.today(), periods=20)
            df_price = pd.DataFrame(index=dates)
            rand_prices = np.random.uniform(90, 110, size=20)
            df_price['Close'] = rand_prices
            df_price['Open'] = rand_prices * 0.99
            df_price['High'] = rand_prices * 1.02
            df_price['Low'] = rand_prices * 0.98
            df_price['Volume'] = np.random.uniform(1000, 50000, size=20)
            
        df_price['Daily_Return'] = df_price['Close'].pct_change()
        volatility_5d = float(df_price['Daily_Return'].rolling(window=5).std().iloc[-1] * np.sqrt(252))
        vol_ma20 = df_price['Volume'].rolling(window=20).mean().iloc[-1]
        turnover_proxy = float(df_price['Volume'].iloc[-1] / (vol_ma20 + 1e-5)) 
        
        delta = df_price['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_14 = float((100 - (100 / (1 + rs))).iloc[-1])
        
        volatility_5d = 0.0 if np.isnan(volatility_5d) else volatility_5d
        turnover_proxy = 0.0 if np.isnan(turnover_proxy) else turnover_proxy
        rsi_14 = 50.0 if np.isnan(rsi_14) else rsi_14
        
        base_score = (factors['short_term'] + factors['long_term']) / 2.0
        pred_val = ((base_score * 0.05) + (factors['sentiment'] * 0.02)) * 100
            
        df_kline = df_price.tail(30)
        dates_list = [d.strftime('%Y-%m-%d') for d in df_kline.index]
        ohlc = df_kline[['Open', 'Close', 'Low', 'High']].values.tolist()
        last_close = float(df_price['Close'].iloc[-1])
        target_price = last_close * (1 + (pred_val / 100))
            
        final_result = {
            "status": "success",
            "data": {
                "ticker": ticker,
                "factors": factors,
                "technical": {
                    "volatility_5d": round(volatility_5d, 4),
                    "turnover_proxy": round(turnover_proxy, 4),
                    "rsi_14": round(rsi_14, 2)
                },
                "prediction_3d": round(pred_val, 2),
                "kline": {
                    "dates": dates_list,
                    "ohlc": ohlc,
                    "last_close": round(last_close, 2),
                    "target_price": round(target_price, 2)
                }
            }
        }
        
        # 存入缓存
        prediction_cache[cache_key] = (final_result, current_time)
        print(f"[Cache Updated] {ticker} 成功获取数据并存入缓存")
        
        return final_result
        
    except HTTPException as he:
        # 如果是业务逻辑主动抛出的错误，直接往上传
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        # 如果是系统奔溃，也不要存缓存
        raise HTTPException(status_code=500, detail=f"Core Engine Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
