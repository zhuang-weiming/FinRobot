import sys
import os
import logging
import pandas as pd
import copy
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
end_date = '20250116'    # 包含预测所需日期
train_start_date = '2022-01-01'
train_end_date = '2024-12-31'
test_start_date = '2025-01-01'
test_end_date = '2025-01-16'
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

# 2.1 计算标准化参数
preprocessor.fit(processed_df)

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
    env_trade = copy.deepcopy(env_trade)
    
    # 获取历史数据的最后一天作为预测起始点
    last_date = env_trade.df.index[-1]
    logger.info(f"历史数据最后一天: {last_date}")
    
    # 生成未来10个交易日
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=10)
    logger.info(f"生成的预测日期: {future_dates}")
    
    future_prices = []
    valid_dates = []

    # 重置环境到历史数据最后一天
    env_trade.current_step = len(env_trade.df) - 1
    logger.info(f"重置环境到第 {env_trade.current_step} 步")

    # 检查环境状态
    if not hasattr(env_trade, 'df') or env_trade.df is None:
        raise ValueError("交易环境数据未正确初始化")
    if not hasattr(env_trade, 'current_step'):
        raise ValueError("交易环境current_step未正确初始化")

    # 进行预测
    for date in future_dates:
        try:
            state = env_trade._get_state()
            action = trained_ddpg_model.predict(state)[0]
            obs, reward, done, info = env_trade.step(action)
            
            # 检查预测结果
            if 'current_price' not in info:
                raise ValueError("预测结果缺少价格信息")
                
            future_prices.append(info['current_price'])
            valid_dates.append(date)
            
            if done:
                logger.warning("环境终止，结束预测")
                break
        except Exception as e:
            logger.error(f"预测步骤出错: {str(e)}")
            raise

    # 创建预测数据DataFrame
    df_predictions = pd.DataFrame({
        'date': valid_dates,
        'price': future_prices
    })
    
    # 确保日期索引正确
    df_predictions['date'] = pd.to_datetime(df_predictions['date'])
    df_predictions.set_index('date', inplace=True)
    
    # 反标准化预测价格
    if hasattr(preprocessor, 'mean_price') and hasattr(preprocessor, 'std_price'):
        df_predictions['price'] = (
            df_predictions['price'] * preprocessor.std_price
        ) + preprocessor.mean_price
        logger.info("已对预测价格进行反标准化处理")
    
    logger.info(f"未来10天预测价格:\n{df_predictions}")

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
    
    # 准备历史数据
    history_df = test_df[['close', 'upper_limit', 'lower_limit']].copy()
    history_df = history_df.rename(columns={'close': 'price'})
    history_df['date'] = history_df.index
    history_df['is_prediction'] = False
    
    # 准备预测数据
    df_predictions = df_predictions.reset_index()
    df_predictions['is_prediction'] = True
    # 预测数据的涨跌停价设置为NaN
    df_predictions['upper_limit'] = np.nan
    df_predictions['lower_limit'] = np.nan
    
    # 合并数据
    combined_df = pd.concat([
        history_df[['date', 'price', 'upper_limit', 'lower_limit', 'is_prediction']],
        df_predictions[['date', 'price', 'upper_limit', 'lower_limit', 'is_prediction']]
    ])
    
    # 确保日期格式一致
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df = combined_df.sort_values('date')
    
    # 可视化预测结果
    plot_stock_predictions(combined_df, history_days=30, predict_days=10, preprocessor=preprocessor)
    logger.info("股票价格预测可视化完成")

except Exception as e:
    logger.error(f"回测失败: {repr(e)}")
    raise
