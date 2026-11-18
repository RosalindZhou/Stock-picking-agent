import yfinance as yf
from google import genai
import json
from pydantic import BaseModel, Field

class QuantFeatures(BaseModel):
    sentiment: float = Field(description="综合情绪得分 (-1.0 到 1.0)")
    short_term: float = Field(description="短期冲击得分 (-1.0 到 1.0)")
    long_term: float = Field(description="长期基本面得分 (-1.0 到 1.0)")
    reason: str = Field(description="一句话核心逻辑")
# 初始化大模型 (请替换真实 Key)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None


def fetch_batch_news(ticker: str, limit: int = 5) -> str:
    """生产级：批量抓取新闻并拼接"""
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news
        if not news_list:
            return ""

        compiled_news = ""
        for i, item in enumerate(news_list[:limit]):
            content = item.get('content', {})
            title = content.get('title', '无标题')
            summary = content.get('summary', '无摘要')
            compiled_news += f"[{i + 1}] 标题：{title} | 摘要：{summary}\n"
        return compiled_news
    except Exception as e:
        print(f"数据抓取异常: {e}")
        return ""


def extract_quant_features(news_text: str) -> dict:
    """生产级：强制大模型使用 Pydantic Schema 输出 JSON 结构化参数"""
    if not news_text or client is None:
        return {"sentiment": 0.0, "short_term": 0.0, "long_term": 0.0, "reason": "无新闻"}

    prompt = f"你是量化分析师。阅读以下新闻，运用你的金融常识，给出一个综合情绪、短期冲击以及长期基本面的量化打分（区间 -1.0 到 1.0，支持浮点），并写出一句话核心逻辑。\n新闻：\n{news_text}"
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
    except Exception as e:
        print(f"Agent JSON 解析异常: {e}")
        return {"sentiment": 0.0, "short_term": 0.0, "long_term": 0.0, "reason": f"解析异常: {str(e)}"}


if __name__ == "__main__":
    # 本地测试
    news = fetch_batch_news("AAPL")
    features = extract_quant_features(news)
    print(features)
import os
