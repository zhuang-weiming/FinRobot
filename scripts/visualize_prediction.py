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

def plot_prediction_results(predictor, train_data, test_data, future_dates, future_predictions, title="上证指数预测结果"):
    """绘制预测结果"""
    feature_columns = ['close', 'volume', 'ma5', 'ma20', 'macd', 'signal', 'rsi', 'volatility']
    predictions = []
    actuals = []
    dates = []
    
    # 合并历史数据
    historical_data = pd.concat([train_data, test_data])
    historical_data.index = pd.to_datetime(historical_data.index)
    
    # 验证时间范围
    assert historical_data.index[0].year >= 2020, f"起始时间应为2020年，当前为{historical_data.index[0].year}年"
    assert historical_data.index[-1].year <= 2025, f"结束时间应为2025年，当前为{historical_data.index[-1].year}年"
    
    # 对历史数据进行滚动预测
    for i in range(0, len(historical_data)-25, 5):
        window = historical_data.iloc[i:i+30]
        features = window[feature_columns].values
        pred = predictor.predict(features)
        actual = historical_data['close'].iloc[i+30] if i+30 < len(historical_data) else None
        
        if actual is not None:
            predictions.append(float(pred[0][0]))
            actuals.append(actual)
            dates.append(historical_data.index[i+30])
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 设置中文字体和样式
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制主图
    plt.subplot(2, 1, 1)
    
    # 绘制完整的历史数据
    plt.plot(historical_data.index, historical_data['close'], 
            label='实际价格', color='#1f77b4', linewidth=2)
    
    # 绘制历史预测
    plt.plot(dates, predictions, 
            label='历史预测', color='#ff7f0e', 
            linestyle='--', linewidth=2)
    
    # 绘制未来预测
    plt.plot(future_dates, future_predictions, 
            label='未来预测', color='#2ca02c', 
            linestyle='--', linewidth=2)
    
    # 添加预测区间
    std_dev = np.std(np.array(predictions) - np.array(actuals))
    plt.fill_between(future_dates, 
                    np.array(future_predictions) - std_dev, 
                    np.array(future_predictions) + std_dev,
                    color='#2ca02c', alpha=0.2,
                    label='预测区间(±1σ)')
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('价格', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴日期格式
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 设置x轴范围
    plt.xlim(pd.Timestamp('2020-01-01'), pd.Timestamp('2025-12-31'))
    
    # 绘制预测误差
    plt.subplot(2, 1, 2)
    errors = np.array(predictions) - np.array(actuals)
    
    # 使用渐变色散点图显示误差
    scatter = plt.scatter(dates, errors, 
                         c=abs(errors), cmap='YlOrRd', 
                         alpha=0.6, s=50)
    plt.colorbar(scatter, label='误差绝对值')
    
    # 添加均值和标准差线
    mean_error = np.mean(errors)
    plt.axhline(y=mean_error, color='red', linestyle='--', 
                alpha=0.8, label=f'平均误差: {mean_error:.2f}')
    plt.axhline(y=mean_error + std_dev, color='gray', linestyle=':',
                alpha=0.5, label=f'±1σ: {std_dev:.2f}')
    plt.axhline(y=mean_error - std_dev, color='gray', linestyle=':',
                alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.title('预测误差分析', fontsize=12)
    plt.xlabel('日期', fontsize=10)
    plt.ylabel('误差', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴日期格式
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 设置x轴范围
    plt.xlim(pd.Timestamp('2020-01-01'), pd.Timestamp('2025-12-31'))
    
    plt.tight_layout()
    return plt.gcf()

def analyze_prediction_performance(predictor, data):
    """分析预测性能和绘制散点图"""
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
            predictions.append(float(pred[0][0]))
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
    plt.style.use('seaborn-v0_8')
    plt.scatter(actuals, predictions, alpha=0.5, color='#1f77b4')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 
            'r--', label='理想预测线')
    plt.title('预测值 vs 实际值', fontsize=14)
    plt.xlabel('实际价格', fontsize=12)
    plt.ylabel('预测价格', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
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

def plot_combined_predictions(predictor, historical_data, future_dates, future_predictions, title="上证指数预测分析"):
    """绘制历史预测和未来预测的组合图"""
    plt.figure(figsize=(15, 12))
    
    # 设置中文字体和样式
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置图表样式
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    
    # 绘制主图
    plt.subplot(2, 1, 1)
    
    # 绘制历史数据（最后3个月）
    last_3m_data = historical_data.iloc[-60:]
    plt.plot(last_3m_data.index, 
            last_3m_data['close'], 
            label='历史价格', 
            color='#1f77b4',
            linewidth=2)
    
    # 绘制未来预测
    plt.plot(future_dates, 
            future_predictions, 
            label='预测价格', 
            color='#ff7f0e',
            linestyle='--', 
            linewidth=2)
    
    # 添加分隔线标记预测起点
    plt.axvline(x=last_3m_data.index[-1], 
                color='gray', 
                linestyle=':', 
                alpha=0.5)
    
    # 添加预测起点标记
    y_range = plt.ylim()
    y_pos = y_range[0] + (y_range[1] - y_range[0]) * 0.02
    plt.text(last_3m_data.index[-1], 
            y_pos,
            '预测起点', 
            rotation=90, 
            verticalalignment='bottom',
            fontsize=10)
    
    plt.title(title, pad=20)
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend(loc='best', fontsize=10)
    
    # 设置x轴日期格式
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # 使用 AutoDateLocator 自动选择合适的日期间隔
    locator = mdates.AutoDateLocator(interval_multiples=True)
    ax.xaxis.set_major_locator(locator)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 添加预测误差子图
    plt.subplot(2, 1, 2)
    
    # 计算预测误差（仅对历史数据部分）
    feature_columns = ['close', 'volume', 'ma5', 'ma20', 'macd', 'signal', 'rsi', 'volatility']
    historical_predictions = []
    actuals = []
    dates = []
    
    for i in range(len(last_3m_data)-30):
        window = last_3m_data.iloc[i:i+30]
        features = window[feature_columns].values
        pred = predictor.predict(features)
        actual = last_3m_data['close'].iloc[i+30]
        
        historical_predictions.append(float(pred[0][0]))
        actuals.append(actual)
        dates.append(last_3m_data.index[i+30])
    
    errors = np.array(historical_predictions) - np.array(actuals)
    
    # 绘制误差柱状图
    plt.bar(dates, errors, color='#2ca02c', alpha=0.6, width=2)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.title('预测误差分析', fontsize=12)
    plt.xlabel('日期', fontsize=10)
    plt.ylabel('误差', fontsize=10)
    
    # 设置x轴日期格式
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # 使用 AutoDateLocator 自动选择合适的日期间隔
    locator = mdates.AutoDateLocator(interval_multiples=True)
    ax.xaxis.set_major_locator(locator)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return plt.gcf()

def main():
    # 加载数据
    config = {
        'stock_code': '000001.SH',
        'start_date': '20200101',
        'end_date': '20241231',
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
    
    # 验证日期范围
    print(f"训练数据时间范围: {train_data.index[0]} 到 {train_data.index[-1]}")
    print(f"测试数据时间范围: {test_data.index[0]} 到 {test_data.index[-1]}")
    
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
    pred_fig = plot_prediction_results(
        predictor,
        train_data,  # 传入训练数据
        test_data,   # 传入测试数据
        future_dates,
        future_predictions,
        "上证指数预测结果 (2020-2025)"
    )
    pred_fig.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    
    # 分析预测性能
    perf_fig = analyze_prediction_performance(predictor, test_data)
    perf_fig.savefig('prediction_performance.png', dpi=300, bbox_inches='tight')
    
    # 打印未来预测结果
    print("\n未来6个月预测结果:")
    for date, pred in zip(future_dates, future_predictions):
        print(f"{date.strftime('%Y-%m-%d')}: {pred:.2f}")
    
    # 保存模型
    predictor.save('stock_predictor.pth')
    print("\n模型已保存到 stock_predictor.pth")

if __name__ == '__main__':
    main() 