import logging
import pandas as pd
import akshare as ak
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ssl
import urllib3
import os
import hashlib
import pickle
from datetime import datetime

# 禁用SSL证书验证
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

def create_session():
    """创建带有重试机制的会话"""
    session = requests.Session()
    retries = Retry(
        total=10,  # 增加重试次数
        backoff_factor=1,  # 增加重试间隔
        status_forcelist=[500, 502, 503, 504, 408, 429],
        allowed_methods=["GET", "POST"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=100,
        pool_maxsize=100
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # 设置超时时间
    session.timeout = 30  # 总超时时间30秒
    
    return session

def get_cache_filename(stock_code, start_date, end_date):
    """生成缓存文件名"""
    cache_key = f"{stock_code}_{start_date}_{end_date}"
    hash_key = hashlib.md5(cache_key.encode()).hexdigest()
    return os.path.join("data_cache", f"{hash_key}.pkl")

def load_stock_data(stock_code, start_date, end_date):
    """
    加载股票历史数据
    
    Args:
        stock_code (str): 需要加载的股票代码 (例如 '601788')。
        start_date (str): 开始日期，格式为YYYYMMDD
        end_date (str): 结束日期，格式为YYYYMMDD
    
    Returns:
        pd.DataFrame: 包含股票历史数据的DataFrame
    """
    if not isinstance(stock_code, str):
        raise TypeError("stock_code must be a string")
    if not stock_code:
        raise ValueError("stock_code cannot be empty")

    # 检查缓存
    cache_file = get_cache_filename(stock_code, start_date, end_date)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                logger.info(f"从缓存加载数据: {cache_file}")
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"缓存加载失败: {str(e)}")

    try:
        # 创建会话
        session = create_session()
        
        # 设置请求头
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
            'Referer': 'https://quote.eastmoney.com/',
            'X-Requested-With': 'XMLHttpRequest',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site'
        }
        
        # 使用akshare获取数据，增加异常处理
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            
            if df.empty:
                logger.warning("获取到的数据为空，尝试备用数据源")
                # 尝试备用数据源
                df = ak.stock_zh_a_spot_em()
                df = df[df['代码'] == stock_code]
                if not df.empty:
                    df = df.iloc[0:1]  # 只取最新数据
                    df['date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
                    df = df.rename(columns={
                        '最新价': 'close',
                        '今开': 'open',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'volume'
                    })
                    df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
                    df['tic'] = stock_code
                    logger.info("成功从备用数据源获取最新数据")
                else:
                    raise ValueError("备用数据源也未获取到数据")
                    
        except Exception as e:
            logger.error(f"获取数据失败: {str(e)}")
            # 尝试从缓存加载
            cache_file = f"data_cache/{stock_code}_{start_date}_{end_date}.csv"
            if os.path.exists(cache_file):
                logger.info(f"从缓存文件 {cache_file} 加载数据")
                df = pd.read_csv(cache_file)
            else:
                raise ValueError(f"无法获取数据且无缓存可用: {str(e)}")
        
        # 数据清洗和格式化
        df["tic"] = stock_code
        df.rename(columns={
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "日期": "date"
        }, inplace=True)
        
        df["date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d')
        df = df[["date", "open", "close", "high", "low", "volume", "tic"]]
        
        logger.info(f"成功加载股票数据，代码: {stock_code}, 记录数: {len(df)}")
        
        # 保存到缓存
        try:
            os.makedirs("data_cache", exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
                logger.info(f"数据已缓存到: {cache_file}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {str(e)}")
            
        return df

    except Exception as e:
        logger.error(f"加载股票数据失败: {str(e)}")
        raise
