import pandas as pd
import yfinance as yf
from sentiment_agent import extract_quant_features

# 模拟历史数据库
historical_events = [
    {"date": "2024-04-24", "ticker": "TSLA", "news": "特斯拉第一季度营收下降9%，但马斯克承诺加速推出平价车型。"},
    {"date": "2024-01-25", "ticker": "TSLA", "news": "特斯拉警告2024年销量增长将大幅放缓。"}
]


def get_actual_return(ticker: str, date_str: str, window_days: int = 5) -> float:
    """生产级：计算未来 N 天的真实收益率 (Label)"""
    try:
        start_date = pd.to_datetime(date_str)
        end_date = start_date + pd.Timedelta(days=window_days + 5)

        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if len(df) >= window_days:
            price_t0 = float(df['Close'].iloc[0])
            price_tn = float(df['Close'].iloc[window_days - 1])
            return round(((price_tn - price_t0) / price_t0) * 100, 2)
    except Exception as e:
        print(f"价格获取异常: {e}")
    return 0.0


if __name__ == "__main__":
    dataset = []
    for event in historical_events:
        features = extract_quant_features(event['news'])
        actual_ret = get_actual_return(event['ticker'], event['date'], window_days=3)

        # 将特征与标签对齐
        row = {
            "Date": event['date'],
            "Ticker": event['ticker'],
            "Agent_Sentiment": features['sentiment'],
            "Agent_ShortTerm": features['short_term'],
            "Actual_Return_3d": actual_ret
        }
        dataset.append(row)

    final_df = pd.DataFrame(dataset)
    print("构建的训练集样本：\n", final_df)
    # 生产环境中，这里会执行 final_df.to_csv("train_data.csv", index=False)
