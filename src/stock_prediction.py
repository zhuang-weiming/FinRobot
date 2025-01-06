import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import timedelta

# 初始化日志记录器
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将 src 目录添加到 Python 的搜索路径
src_dir = os.path.join(current_dir, '..')
sys.path.append(src_dir)

from src.data_loader import load_stock_data
from src.preprocessor import DataPreprocessor, split_data
from src.single_stock_env import create_single_stock_env
from src.agent import create_and_train_agent, predict_with_agent
from src.backtest import plot_stock_predictions

# 初始化数据预处理器
preprocessor = DataPreprocessor()

# 配置参数
stock_code = '601788'  # 光大证券
start_date = '20220101'  # 扩展数据范围
end_date = '20231231'    # 使用真实历史数据
train_start_date = '2022-01-01'
train_end_date = '2022-12-31'
test_start_date = '2023-01-01'
test_end_date = '2023-06-30'
tech_indicator_list = ["macd", "rsi", "cci", "dx"]
initial_amount = 1000000
hmax = 100
buy_cost_pct = 0.001
sell_cost_pct = 0.001
reward_scaling = 1e-4
batch_size = 64
buffer_size = 100000
learning_rate = 0.0001
net_arch = [512, 512]
total_timesteps = 10000
reward_func = "SharpeRatio"

# 1. 加载数据
raw_df = load_stock_data(stock_code, start_date, end_date)

# 使用原始日期格式
raw_df.index = pd.to_datetime(raw_df.index)

# 2. 数据预处理和特征工程
processed_df = preprocessor.preprocess_data(raw_df, tech_indicator_list=tech_indicator_list)

# 3. 标准化数据
processed_df = preprocessor.standardize(processed_df)

# 4. 划分训练集和测试集
train_df, test_df = split_data(processed_df, train_start_date, train_end_date, test_start_date, test_end_date)

try:
    # 5. 创建交易环境
    logger.info("开始创建交易环境...")
    env_train, env_trade, stock_dimension, state_space = create_single_stock_env(
        train_df=train_df,
        test_df=test_df,
        processed_df=processed_df,
        tech_indicator_list=tech_indicator_list,
        initial_amount=initial_amount,
        hmax=hmax,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        reward_scaling=reward_scaling,
        reward_func=reward_func,
    )
    logger.info(f"交易环境创建成功，股票数量: {stock_dimension}, 状态空间: {state_space}")
except Exception as e:
    logger.error(f"预测失败: {repr(e)}")
    raise

try:
    # 6. 创建和训练 Agent
    logger.info("开始训练Agent...")
    trained_ddpg_model = create_and_train_agent(
        env_train=env_train,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_rate=learning_rate,
        net_arch=net_arch,
        total_timesteps=total_timesteps,
    )
    logger.info("Agent训练完成")
except Exception as e:
    logger.error(f"Agent训练失败: {repr(e)}")
    raise

try:
    # 7. 使用训练好的 Agent 进行未来10天预测
    logger.info("开始使用训练好的Agent进行未来10天预测...")
    
    # 调试信息：打印env_trade对象类型和属性
    logger.info(f"env_trade类型: {type(env_trade)}")
    logger.info(f"env_trade属性: {dir(env_trade)}")
    
    # 获取测试环境的初始状态
    state = env_trade.reset()
    
    future_prices = []
    
    for _ in range(10):  # 预测未来10天
        action = trained_ddpg_model.predict(state)
        
        # 执行一个虚拟步骤以获取下一个状态和预测价格
        obs, reward, terminated, info = env_trade.step(action)
        
        # 假设 info 中包含了当天的价格信息
        if 'current_price' in info:
            future_prices.append(info['current_price'])
        else:
            logger.warning("环境中没有提供当前价格信息。")
            break
        
        state = obs
        
        if terminated:
            break
    
    if future_prices:
        # 确保test_df.index是datetime格式并正确转换
        test_df.index = pd.to_datetime(test_df.index, format='%Y-%m-%d', errors='coerce').floor('D')
        # 验证日期格式
        if test_df.index.isnull().any():
            logger.error("日期格式转换失败，请检查数据源")
            raise ValueError("日期格式转换失败")
        # 获取最后一个有效日期
        last_date = test_df.index[-1]
        logger.info(f"最后一个有效日期: {last_date}")
        # 生成未来日期序列，使用正确的基准日期
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=len(future_prices),
            freq='B'  # 只生成工作日
        )
        # 创建预测结果DataFrame
        df_future_prices = pd.DataFrame({
            'date': future_dates,
            'price': future_prices
        })
        logger.info(f"未来10天预测价格:\n{df_future_prices}")
    else:
        logger.warning("未能获取到未来价格预测。")

except Exception as e:
    logger.error(f"预测失败: {str(e)}")
    raise

try:
    # 8. 回测和可视化
    logger.info("开始回测...")

    # 使用测试集进行回测
    df_results, df_actions = predict_with_agent(
        agent=trained_ddpg_model,
        environment=env_trade,
        deterministic=True
    )

    # 确保df_results包含price列
    if 'current_price' not in df_results.columns:
        raise ValueError("回测数据缺少price列")
    
    # 重命名current_price为price
    df_results = df_results.rename(columns={'current_price': 'price'})
    
    # 确保日期格式正确
    df_results['date'] = pd.to_datetime(df_results.index)
    
    # 创建回测数据DataFrame
    df_predictions = df_results[['date', 'price']].copy()

    # 可视化预测结果
    history_days = 30  # 显示最近30天历史数据
    predict_days = 10  # 显示未来10天预测
    plot_stock_predictions(df_predictions, history_days, predict_days)
    logger.info("股票价格预测可视化完成")

except Exception as e:
    logger.error(f"回测失败: {repr(e)}")
    raise
