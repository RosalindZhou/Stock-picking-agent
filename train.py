import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

print("正在初始化量化训练引擎...")

# ==========================================
# 1. 模拟你的真实历史数据集 (未来替换成你用大模型跑出来的真实 CSV)
# ==========================================
# 在真实业务中，你需要用之前的 backtest_engine 跑出至少几千条数据，这里我们用 Numpy 模拟
np.random.seed(42)
num_samples = 2000

# 模拟大模型给出的因子 (-1.0 到 1.0)
agent_sentiment = np.random.uniform(-1, 1, num_samples)
agent_short_term = np.random.uniform(-1, 1, num_samples)
agent_long_term = np.random.uniform(-1, 1, num_samples)

# 模拟基础技术面因子 (比如过去5天的波动率、换手率)
volatility_5d = np.random.uniform(0.01, 0.1, num_samples)
turnover_ratio = np.random.uniform(0.001, 0.05, num_samples)

# 构造目标变量 y：真实的 T+3 收益率 (带一点随机噪音)
# 假设逻辑：短期分数越高，真实收益率越倾向于高；波动率越高，收益率方差越大
actual_return_3d = (agent_short_term * 0.03) + (agent_sentiment * 0.01) + np.random.normal(0, volatility_5d)

df = pd.DataFrame({
    'Agent_Sentiment': agent_sentiment,
    'Agent_ShortTerm': agent_short_term,
    'Agent_LongTerm': agent_long_term,
    'Volatility_5d': volatility_5d,
    'Turnover': turnover_ratio,
    'Target_Return_3d': actual_return_3d
})

print(f"成功加载数据集，样本量: {len(df)}")

# ==========================================
# 2. 特征与标签分离，切分训练集与测试集
# ==========================================
features = ['Agent_Sentiment', 'Agent_ShortTerm', 'Agent_LongTerm', 'Volatility_5d', 'Turnover']
X = df[features]
y = df['Target_Return_3d']

# 按照 8:2 切分（在真实量化中，必须按时间顺序切分避免未来函数，这里简化处理）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 LightGBM 专属数据格式（极大地提升训练速度）
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# ==========================================
# 3. 设置 LightGBM 超参数并训练
# ==========================================
params = {
    'objective': 'regression',       # 预测具体的涨跌幅百分比
    'metric': 'rmse',                # 评估指标：均方根误差
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,                # 控制树的复杂度，防止过拟合
    'feature_fraction': 0.8,         # 每次迭代随机抽取 80% 的特征
    'verbose': -1
}

print("开始训练 LightGBM 多模态融合模型...")

# 使用回调函数替代旧版的 early_stopping_rounds
callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True),
             lgb.log_evaluation(period=10)]

gbm = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[test_data],
    callbacks=callbacks
)

# ==========================================
# 4. 模型评估与特征重要性分析
# ==========================================
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"✅ 模型训练完成！测试集 RMSE (均方根误差): {rmse:.4f}")

# 保存模型到本地（这是最重要的产出物！）
model_path = 'lgb_quant_model.txt'
gbm.save_model(model_path)
print(f"📦 模型已保存至: {os.path.abspath(model_path)}")

# 打印特征重要性 (看大模型的情绪因子到底有没有用)
importance = gbm.feature_importance()
for feature, imp in zip(features, importance):
    print(f"因子 [{feature}] 的重要性得分: {imp}")
