import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
from src.data.data_loader import StockDataLoader
from src.models.prediction_model import PredictionManager

def run_prediction_experiment(stock_code: str):
    """运行预测实验"""
    # 1. 配置参数
    config = {
        'stock_code': stock_code,
        'start_date': '20200101',    # 从2020年开始
        'end_date': '20241231',      # 到2024年底
        'train_ratio': 0.8,          # 80%用于训练
    }
    
    print(f"\n开始 {StockDataLoader.SUPPORTED_STOCKS.get(stock_code, stock_code)} 的预测实验...")
    
    # 2. 加载数据
    data_loader = StockDataLoader(stock_code, config)
    train_data, test_data = data_loader.load_and_split_data(
        config['start_date'],
        config['end_date'],
        config['train_ratio']
    )
    
    print(f"\n数据加载完成:")
    print(f"训练数据: {train_data.index[0]} 到 {train_data.index[-1]}")
    print(f"测试数据: {test_data.index[0]} 到 {test_data.index[-1]}")
    
    # 3. 创建和训练模型
    predictor = PredictionManager(
        lookback_window=30,      # 使用30天数据
        predict_window=5,        # 预测未来5天
        batch_size=64,
        learning_rate=1e-4
    )
    
    print("\n开始训练模型...")
    history = predictor.train(train_data, test_data, epochs=50, verbose=True)
    
    # 4. 生成预测
    # 使用最后30天数据预测未来
    latest_data = test_data.iloc[-30:].copy()
    future_dates = pd.date_range(
        start=test_data.index[-1] + pd.Timedelta(days=1),
        periods=30,
        freq='B'  # 使用工作日
    )
    
    feature_columns = ['close', 'volume', 'ma5', 'ma20', 'macd', 'signal', 'rsi', 'volatility']
    features = latest_data[feature_columns].values
    predictions = []
    
    # 滚动预测未来30天
    for _ in range(len(future_dates)):
        pred = predictor.predict(features)
        predictions.append(float(pred[0][0]))
        
        # 更新特征用于下一次预测
        features = np.roll(features, -1, axis=0)
        features[-1] = features[-2]  # 复制前一天的特征
        features[-1, 0] = predictions[-1]  # 更新收盘价
    
    # 5. 计算预测准确性指标（对测试集）
    test_predictions = []
    for i in range(len(test_data)-30):
        features = test_data[feature_columns].iloc[i:i+30].values
        pred = predictor.predict(features)
        test_predictions.append(float(pred[0][0]))
    
    test_actuals = test_data['close'].iloc[30:].values
    mse = np.mean((np.array(test_predictions) - test_actuals) ** 2)
    mae = np.mean(np.abs(np.array(test_predictions) - test_actuals))
    mape = np.mean(np.abs((np.array(test_predictions) - test_actuals) / test_actuals)) * 100
    
    print("\n预测性能评估:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # 6. 打印未来预测结果
    print("\n未来30天预测结果:")
    for date, pred in zip(future_dates, predictions):
        print(f"{date.strftime('%Y-%m-%d')}: {pred:.2f}")
    
    # 7. 保存模型
    model_path = f'{stock_code}_predictor.pth'
    predictor.save(model_path)
    print(f"\n模型已保存到 {model_path}")
    
    return predictor, history, predictions, future_dates, test_data, test_predictions, test_actuals

def main():
    parser = argparse.ArgumentParser(description='股票预测实验')
    parser.add_argument('--stock', type=str, default='000001.SH',
                      choices=['000001.SH', '300059.SZ', '601788.SH'],
                      help='股票代码 (000001.SH-上证指数, 300059.SZ-东方财富, 601788.SH-光大证券)')
    args = parser.parse_args()
    
    # 添加结果可视化，接收所有返回值
    predictor, history, predictions, future_dates, test_data, test_predictions, test_actuals = run_prediction_experiment(args.stock)
    
    # 绘制预测结果
    plt.figure(figsize=(15, 10))
    plt.style.use('seaborn-v0_8')
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制历史数据和预测
    stock_names = {
        '000001.SH': '上证指数',
        '300059.SZ': '东方财富',
        '601788.SH': '光大证券'
    }
    
    plt.title(f"{stock_names[args.stock]}预测结果", fontsize=14, pad=20)
    plt.plot(test_data.index, test_data['close'], label='历史数据', color='blue')
    plt.plot(test_data.index[30:], test_predictions, label='历史预测', color='green', linestyle='--')
    plt.plot(future_dates, predictions, label='未来预测', color='red', linestyle='--')
    
    # 添加预测区间
    std_dev = np.std(np.array(test_predictions) - test_actuals)
    plt.fill_between(future_dates,
                    np.array(predictions) - std_dev,
                    np.array(predictions) + std_dev,
                    color='red', alpha=0.2,
                    label='预测区间(±1σ)')
    
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('价格', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    # 保存图表
    plt.savefig(f'{args.stock}_prediction.png', dpi=300, bbox_inches='tight')
    print(f"\n预测结果图表已保存到 {args.stock}_prediction.png")

if __name__ == '__main__':
    main() 