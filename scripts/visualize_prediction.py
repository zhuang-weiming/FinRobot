import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.data.data_loader import StockDataLoader
from src.models.prediction_model import PredictionManager
import seaborn as sns

def plot_prediction_results(predictor, data, title="股票预测结果"):
    """绘制预测结果"""
    feature_columns = ['close', 'volume', 'ma5', 'ma20', 'macd', 'signal', 'rsi', 'volatility']
    predictions = []
    actuals = []
    dates = []
    
    # 滚动预测
    for i in range(0, len(data)-25, 5):
        window = data.iloc[i:i+30]
        features = window[feature_columns].values
        pred = predictor.predict(features)
        actual = data['close'].iloc[i+30] if i+30 < len(data) else None
        
        if actual is not None:
            predictions.append(float(pred))
            actuals.append(actual)
            dates.append(data.index[i+30])
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 设置日期格式
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 绘制实际价格和预测价格
    plt.subplot(2, 1, 1)
    plt.plot(dates, actuals, label='实际价格', color='blue')
    plt.plot(dates, predictions, label='预测价格', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    
    # 设置x轴日期格式
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # 自动选择合适的日期间隔
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # 旋转45度并右对齐
    
    # 绘制预测误差
    plt.subplot(2, 1, 2)
    errors = np.array(predictions) - np.array(actuals)
    plt.bar(dates, errors, color='green', alpha=0.6)
    plt.title('预测误差')
    plt.xlabel('日期')
    plt.ylabel('误差')
    plt.grid(True)
    
    # 设置x轴日期格式
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # 自动选择合适的日期间隔
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # 旋转45度并右对齐
    
    plt.tight_layout()  # 调整子图间距
    return plt.gcf()

def analyze_prediction_performance(predictor, data):
    """分析预测性能"""
    feature_columns = ['close', 'volume', 'ma5', 'ma20', 'macd', 'signal', 'rsi', 'volatility']
    predictions = []
    actuals = []
    
    # 滚动预测
    for i in range(0, len(data)-25, 5):
        window = data.iloc[i:i+30]
        features = window[feature_columns].values
        pred = predictor.predict(features)
        actual = data['close'].iloc[i+30] if i+30 < len(data) else None
        
        if actual is not None:
            predictions.append(float(pred))
            actuals.append(actual)
    
    # 计算性能指标
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
    
    print("\n预测性能分析:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # 绘制预测vs实际散点图
    plt.figure(figsize=(10, 10))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.title('预测值 vs 实际值')
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.grid(True)
    
    return plt.gcf()

def predict_future(predictor, last_data, future_dates):
    """预测未来数据"""
    feature_columns = ['close', 'volume', 'ma5', 'ma20', 'macd', 'signal', 'rsi', 'volatility']
    features = last_data[feature_columns].values
    
    future_predictions = []
    for _ in range(len(future_dates)):
        pred = predictor.predict(features)
        future_predictions.append(float(pred[0][0]))  # 提取预测值
        
        # 更新特征，将预测值加入到特征中
        features = np.roll(features, -1, axis=0)
        features[-1] = features[-2]  # 复制前一天的所有特征
        features[-1, 0] = float(pred[0][0])  # 更新最后一行的收盘价
        
        # 更新技术指标
        features[-1, 2] = np.mean(features[-5:, 0])  # 更新MA5
        features[-1, 3] = np.mean(features[-20:, 0])  # 更新MA20
        
    return future_predictions

def plot_prediction_with_future(historical_data, future_dates, future_predictions, title="上证指数预测结果"):
    """绘制历史数据和未来预测"""
    plt.figure(figsize=(15, 10))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制历史数据（最后60天）
    plt.plot(historical_data.index[-60:], 
            historical_data['close'].iloc[-60:], 
            label='历史价格', 
            color='blue')
    
    # 绘制预测数据
    plt.plot(future_dates, 
            future_predictions, 
            label='预测价格', 
            color='red', 
            linestyle='--')
    
    plt.title(title)
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    
    # 设置x轴日期格式
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # 每月显示一个刻度
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return plt.gcf()

def main():
    # 加载数据
    config = {
        'stock_code': '000001.SH',
        'start_date': '20200101',
        'end_date': '20241231',  # 训练数据截止到2024年底
        'train_ratio': 0.8
    }
    
    data_loader = StockDataLoader(config['stock_code'], config)
    train_data, test_data = data_loader.load_and_split_data(
        config['start_date'],
        config['end_date'],
        config['train_ratio']
    )
    
    # 确保日期格式正确
    train_data.index = pd.to_datetime(train_data.index)
    test_data.index = pd.to_datetime(test_data.index)
    
    # 创建和训练模型
    predictor = PredictionManager(
        lookback_window=20,
        predict_window=5,
        batch_size=64,
        learning_rate=1e-4
    )
    
    print("训练模型中...")
    history = predictor.train(train_data, test_data, epochs=50, verbose=True)
    
    # 生成未来日期序列
    future_dates = pd.date_range(start='2025-01-25', end='2025-07-25', freq='5D')
    
    # 使用最后30天数据进行未来预测
    latest_data = test_data.iloc[-30:].copy()
    future_predictions = predict_future(predictor, latest_data, future_dates)
    
    # 绘制预测结果
    pred_fig = plot_prediction_with_future(
        test_data, 
        future_dates, 
        future_predictions, 
        "上证指数预测结果 (2025年1月-7月)"
    )
    pred_fig.savefig('future_prediction_results.png')
    
    # 打印预测结果
    print("\n未来6个月预测结果:")
    for date, pred in zip(future_dates, future_predictions):
        print(f"{date.strftime('%Y-%m-%d')}: {pred:.2f}")
    
    # 保存模型
    predictor.save('stock_predictor.pth')
    print("\n模型已保存到 stock_predictor.pth")

if __name__ == '__main__':
    main() 