import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import re
from google import genai
from google.genai import errors
import json
from pydantic import BaseModel, Field

class QuantFeatures(BaseModel):
    sentiment: float = Field(description="综合情绪得分 (-1.0 到 1.0)")
    short_term: float = Field(description="短期冲击得分 (-1.0 到 1.0)")
    long_term: float = Field(description="长期基本面得分 (-1.0 到 1.0)")
    reason: str = Field(description="一句话核心逻辑概述")

# ================= Configuration =================
# 安全读取本地 .env 保护密钥
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ[k] = v

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

SYMBOL = "600519"  # 演示沪深 300 标的 (A股)
DAYS_TO_FETCH = 365
OUTPUT_FILE = f"historical_factors_{SYMBOL.lower()}.csv"

# Rate limits handling
FINNHUB_RPM = 60
GEMINI_RPM = 15  # Free tier is 15 requests per minute
GEMINI_SLEEP_SEC = 60.0 / GEMINI_RPM + 0.5  # Add a slight buffer

client = None
if GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY":
    client = genai.Client(api_key=GOOGLE_API_KEY)


def init_csv():
    """初始化 CSV 表头，如果不存在则创建。返回已处理的新闻 ID 集合"""
    headers = [
        "Date", "Ticker", "News_ID", "Title", "Summary",
        "Agent_Sentiment", "Agent_ShortTerm", "Agent_LongTerm", "Reason"
    ]
    if not os.path.exists(OUTPUT_FILE):
        df = pd.DataFrame(columns=headers)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        return set()
    else:
        try:
            df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
            return set(df["News_ID"].astype(str).tolist())
        except Exception:
            # 如果解析失败，可能是空文件
            return set()


def fetch_news(symbol: str, from_date: str, to_date: str) -> list:
    """智能抓取一段日期内的公司新闻 (美股Finnhub，A股AkShare)"""
    if symbol.isdigit() or symbol.endswith(".SS") or symbol.endswith(".SZ"):
        pure_symbol = symbol.split(".")[0]
        try:
            import akshare as ak
            df_news = ak.stock_news_em(symbol=pure_symbol)
            items = []
            if df_news is not None and not df_news.empty:
                for idx, row in df_news.iterrows():
                    title = str(row.get('新闻标题', ''))
                    content = str(row.get('新闻内容', ''))
                    dt_str = str(row.get('发布时间', ''))
                    items.append({
                        "id": f"cn_{idx}_{dt_str}",
                        "headline": title,
                        "summary": content,
                        "datetime": 0,
                        "dt_str": dt_str
                    })
            return items
        except Exception as e:
            print(f"AkShare 数据抓取异常 ({symbol}): {e}")
            return []
    else:
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 429:
                print("Finnhub 频控限制，等待 60 秒...")
                time.sleep(60)
                return fetch_news(symbol, from_date, to_date)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Finnhub 数据抓取异常 ({from_date} to {to_date}): {e}")
            return []


def extract_quant_features_with_retry(news_text: str, retries: int = 3) -> dict:
    """生产级：强制大模型使用 Pydantic Schema 输出结构化参数，带重试退避机制"""
    if not client:
        return {"sentiment": 0.0, "short_term": 0.0, "long_term": 0.0, "reason": "未配置大模型 Key"}
        
    if not news_text or len(news_text.strip()) < 5:
        return {"sentiment": 0.0, "short_term": 0.0, "long_term": 0.0, "reason": "无新闻或文本过短"}

    prompt = f"你是量化分析师。阅读以下新闻，运用你的金融常识，给出一个综合情绪、短期冲击以及长期基本面的量化打分（区间 -1.0 到 1.0，支持浮点），并写出一句话核心逻辑。\n新闻资讯：\n{news_text}"

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite', 
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': QuantFeatures
                }
            )
            
            data = json.loads(response.text)
            return {
                "sentiment": float(data.get("sentiment", 0.0)),
                "short_term": float(data.get("short_term", 0.0)),
                "long_term": float(data.get("long_term", 0.0)),
                "reason": str(data.get("reason", "解析成功"))
            }
        except errors.APIError as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait_time = (2 ** attempt) * 10
                print(f"Gemini 频控限制 (429)，退避重试，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                print(f"Gemini APIError 异常: {e}")
                time.sleep(5)
        except Exception as e:
            print(f"Agent JSON 解析异常: {e}")
            time.sleep(5)
            
    return {"sentiment": 0.0, "short_term": 0.0, "long_term": 0.0, "reason": "多次尝试解析均失败"}


def main():
    if FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY" or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        print("请注意：您正在使用默认的 API Key 占位符。请确保在环境变量中配置了 FINNHUB_API_KEY 和 GOOGLE_API_KEY。")
        print("如果不配置真实 Key，抓取或打分将会失败。部分功能仍然可以运行演示。")

    processed_ids = init_csv()
    print(f"已发现本地归档数据，共 {len(processed_ids)} 条已处理记录。支持断点续传。")

    end_date_obj = datetime.now()
    start_date_obj = end_date_obj - timedelta(days=DAYS_TO_FETCH)
    
    # 按照 30 天为一个 chunk 向前抓取，避免单次返回过多以及 Finnhub 限制
    chunk_days = 30
    current_end = end_date_obj
    
    total_new_saved = 0

    while current_end > start_date_obj:
        current_start = current_end - timedelta(days=chunk_days)
        if current_start < start_date_obj:
            current_start = start_date_obj
            
        from_str = current_start.strftime('%Y-%m-%d')
        to_str = current_end.strftime('%Y-%m-%d')
        
        print(f"正在抓取区间: {from_str} 到 {to_str}")
        news_items = fetch_news(SYMBOL, from_str, to_str)
        
        if news_items:
            print(f"获取到 {len(news_items)} 条新闻。准备进入 AI 因子抽取环节...")
        
        for item in news_items:
            news_id = str(item.get("id", ""))
            
            # 断点续传过滤去重
            if news_id in processed_ids or not news_id:
                continue
                
            dt_timestamp = item.get("datetime")
            news_date = item.get("dt_str") if "dt_str" in item else (datetime.fromtimestamp(dt_timestamp).strftime('%Y-%m-%d %H:%M:%S') if dt_timestamp else to_str)
            title = item.get("headline", "").replace("\n", " ").replace("\r", " ")
            summary = item.get("summary", "").replace("\n", " ").replace("\r", " ")
            
            combined_text = f"标题：{title} \n摘要：{summary}"
            
            # 若标题和摘要都为空，直接跳过
            if not title and not summary:
                continue

            print(f"[{news_date}] 正在处理: {title[:30]}...")
            
            # AI 评分
            features = extract_quant_features_with_retry(combined_text)
            
            row_dict = {
                "Date": news_date,
                "Ticker": SYMBOL,
                "News_ID": news_id,
                "Title": title,
                "Summary": summary,
                "Agent_Sentiment": features['sentiment'],
                "Agent_ShortTerm": features['short_term'],
                "Agent_LongTerm": features['long_term'],
                "Reason": features['reason']
            }
            
            # 增量落盘
            df_row = pd.DataFrame([row_dict])
            df_row.to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding="utf-8-sig")
            
            processed_ids.add(news_id)
            total_new_saved += 1
            
            # 等待，遵守 Gemini 频控 (Free Tier 15 RPM)
            time.sleep(GEMINI_SLEEP_SEC)
            
        # Finnhub 可能会有限制，且为稳妥起见，每个 chunk 抓好歇一下
        time.sleep(2)
        
        current_end = current_start
        
    print(f"\n全部任务完成！本次运行共新增收割数据 {total_new_saved} 条。文件保存在: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
