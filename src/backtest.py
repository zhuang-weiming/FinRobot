import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def validate_account_value_data(df):
    """验证账户价值数据格式"""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("输入数据必须是Pandas DataFrame")
    
    # 处理环境类返回的数据
    if 'current_price' in df.columns and 'date' in df.columns:
        df = df.rename(columns={'current_price': 'price'})
    elif 'predicted_price' in df.columns and 'date' in df.columns:
        df = df.rename(columns={'predicted_price': 'price'})
    elif 'price' in df.columns and 'date' in df.columns:
        # 已经是正确的列名，直接处理
        pass
    else:
        # 处理预测数据格式
        if len(df.columns) == 2:
            df.columns = ['date', 'price']
    
    required_columns = {'price', 'date'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"输入数据必须包含以下列: {required_columns}")
    
    if df.empty:
        raise ValueError("输入数据不能为空")
    
    # 处理日期列
    if 'date' in df.index.names and 'date' in df.columns:
        # 如果date同时存在于索引和列中，删除索引中的date
        df = df.reset_index(drop=True)
    elif 'date' in df.index.names:
        if 'date' not in df.columns:
            df = df.reset_index()
    
    # 确保date列是pandas Series
    if isinstance(df['date'], np.ndarray):
        df['date'] = pd.Series(df['date'])
    
    # 预处理日期列
    if isinstance(df['date'], np.ndarray):
        # 如果日期列是numpy数组，先转换为Series
        df['date'] = pd.Series(df['date'])
    
    # 确保日期列是字符串类型
    df['date'] = df['date'].astype(str)
    
    # 转换日期格式
    try:
        # 尝试多种日期格式
        df['date'] = pd.to_datetime(
            df['date'],
            format='mixed',  # 支持多种日期格式
            errors='coerce'
        )
        
        # 检查转换结果
        if df['date'].isnull().any():
            # 尝试修复常见日期格式问题
            df['date'] = pd.to_datetime(
                df['date'].fillna('1970-01-01'),
                format='%Y-%m-%d',
                errors='coerce'
            )
            
            if df['date'].isnull().any():
                raise ValueError("日期列包含无法转换的无效值")
                
    except Exception as e:
        logger.error(f"日期列转换失败: {str(e)}")
        raise ValueError(f"日期列转换失败: {str(e)}")
    # 处理可能的重复日期
    df = df.drop_duplicates(subset=['date'], keep='last')
    
    # 数据排序
    df = df.sort_values('date').reset_index(drop=True)
    
    # 处理price列
    try:
        # 确保price列是数值类型
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # 处理缺失值
        if df['price'].isnull().any():
            logger.warning("发现price存在缺失值，将进行填充处理")
            df['price'] = df['price'].ffill()
            
        # 确保price列是Series类型
        if not isinstance(df['price'], pd.Series):
            df['price'] = pd.Series(df['price'])
            
        # 验证price列数据
        if df['price'].isnull().any():
            raise ValueError("price列包含无法转换的无效值")
            
    except Exception as e:
        logger.error(f"无法处理price列: {str(e)}")
        raise ValueError("price列必须是可转换为数值类型的一维数据")
    
    return df

def calculate_metrics(df):
    """计算回测指标"""
    try:
        df = validate_account_value_data(df)
        
        # 计算收益率
        if isinstance(df['price'], pd.Series):
            df['returns'] = df['price'].pct_change(fill_method=None)
        else:
            df = df.copy()
            df['returns'] = df['price'].iloc[:, 0].pct_change(fill_method=None)
            
        # 确保收益率是数值类型
        df['returns'] = pd.to_numeric(df['returns'], errors='coerce')
        df['returns'] = df['returns'].fillna(0)
        
        # 确保价格数据是数值类型
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['price'] = df['price'].fillna(0)
        
        # 计算总收益率（确保返回标量值）
        total_return = float(df['price'].iloc[-1].item() / df['price'].iloc[0].item() - 1)
        
        # 计算夏普比率（确保返回标量值）
        sharpe_ratio = float(df['returns'].mean().item() / df['returns'].std().item()) if df['returns'].std().item() != 0 else np.nan
        
        # 计算最大回撤（确保返回标量值）
        peak = df['price'].cummax()
        drawdown = (peak - df['price']) / peak
        max_drawdown = float(drawdown.max().item())
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        logger.info(f"回测指标计算完成: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"回测指标计算失败: {str(e)}")
        raise

def plot_stock_predictions(df_predictions, history_days=30, predict_days=10):
    """
    可视化股票价格预测
    :param df_predictions: 预测数据，包含date和price两列
    :param history_days: 显示的历史天数
    :param predict_days: 预测的天数
    """
    try:
        # 数据验证
        if not isinstance(df_predictions, pd.DataFrame):
            raise ValueError("输入数据必须是Pandas DataFrame")
            
        if not {'date', 'price'}.issubset(df_predictions.columns):
            raise ValueError("输入数据必须包含date和price两列")
            
        # 确保日期格式正确
        df_predictions['date'] = pd.to_datetime(df_predictions['date'])
        
        # 设置绘图
        plt.figure(figsize=(15, 6))
        
        # 计算日期范围
        end_date = df_predictions['date'].max()
        start_date = end_date - timedelta(days=history_days)
        future_end_date = end_date + timedelta(days=predict_days)
        
        # 过滤数据
        filtered_df = df_predictions[(df_predictions['date'] >= start_date) & 
                                   (df_predictions['date'] <= future_end_date)]
        
        # 绘制历史价格曲线
        plt.plot(filtered_df[filtered_df['date'] <= end_date]['date'], 
                filtered_df[filtered_df['date'] <= end_date]['price'],
                color='blue', linestyle='-', label='历史价格')
        
        # 绘制预测价格曲线
        plt.plot(filtered_df[filtered_df['date'] > end_date]['date'], 
                filtered_df[filtered_df['date'] > end_date]['price'],
                color='red', linestyle='--', label='预测价格')
        
        # 设置坐标轴格式
        plt.gca().yaxis.set_major_formatter('¥{x:,.2f}')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, predict_days//5)))
        
        # 添加预测区间标记
        plt.axvline(x=end_date, color='green', linestyle=':', label='预测起始点')
        
        # 设置图表属性
        plt.xlabel('日期')
        plt.ylabel('股票价格 (¥)')
        plt.title('光大证券股票价格预测')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 显示图表
        plt.show(block=False)
        plt.pause(0.1)
        logger.info("股票价格预测可视化完成，等待窗口关闭...")
        plt.waitforbuttonpress()
        logger.info("窗口已关闭，继续执行程序")
        
    except Exception as e:
        logger.error(f"股票价格预测可视化失败: {str(e)}")
        raise

if __name__ == '__main__':
    # 测试数据
    data = {
        'date': pd.date_range(start='2023-01-01', periods=100),
        'price': np.cumprod(1 + np.random.normal(0, 0.01, 100)) * 100
    }
    df_predictions = pd.DataFrame(data)
    
    # 测试可视化
    plot_stock_predictions(df_predictions)
