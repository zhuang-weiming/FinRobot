import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime, timedelta

class FinGPTSentimentAnalyzer:
    """金融文本情感分析器"""
    
    def __init__(self, model_name: str = "bert-base-chinese", device: str = None):
        """
        初始化情感分析器
        
        Args:
            model_name: 预训练模型名称
            device: 计算设备 ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # 加载分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3  # 负面、中性、正面
            ).to(self.device)
            
            # 设置标签映射
            self.label_map = {0: "负面", 1: "中性", 2: "正面"}
            
        except Exception as e:
            logging.error(f"Error initializing FinGPT model: {str(e)}")
            raise
    
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        分析单条文本
        
        Args:
            text: 输入文本
            
        Returns:
            情感分析结果
        """
        try:
            # 文本预处理
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                
            # 获取预测结果
            label_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][label_id].item()
            
            # 计算情感得分 (-1 到 1)
            sentiment_score = (label_id - 1) * confidence
            
            return {
                "label": self.label_map[label_id],
                "confidence": confidence,
                "sentiment_score": sentiment_score
            }
            
        except Exception as e:
            logging.error(f"Error analyzing text: {str(e)}")
            return {
                "label": "中性",
                "confidence": 0.0,
                "sentiment_score": 0.0
            }
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """批量分析文本"""
        results = []
        
        try:
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i + batch_size]
                
                # 批量编码
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # 批量推理
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                
                # 处理每条结果
                for j in range(len(batch_texts)):
                    label_id = torch.argmax(probs[j]).item()
                    confidence = probs[j][label_id].item()
                    sentiment_score = (label_id - 1) * confidence
                    
                    results.append({
                        "text": batch_texts[j],
                        "label": self.label_map[label_id],
                        "confidence": confidence,
                        "sentiment_score": sentiment_score
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Error in batch analysis: {str(e)}")
            return []
    
    def analyze_news_impact(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        分析新闻影响
        
        Args:
            news_df: 包含新闻数据的DataFrame
            
        Returns:
            添加了情感分析结果的DataFrame
        """
        try:
            # 提取需要分析的文本
            texts = news_df['title'].tolist()  # 假设新闻标题在'title'列
            
            # 批量分析
            results = self.analyze_batch(texts)
            
            # 将结果添加到DataFrame
            news_df['sentiment_label'] = [r['label'] for r in results]
            news_df['sentiment_score'] = [r['sentiment_score'] for r in results]
            news_df['confidence'] = [r['confidence'] for r in results]
            
            # 计算每日聚合情感得分
            daily_sentiment = news_df.groupby('date').agg({
                'sentiment_score': ['mean', 'std', 'count'],
                'confidence': 'mean'
            }).reset_index()
            
            # 重命名列
            daily_sentiment.columns = [
                'date', 'avg_sentiment', 'sentiment_std',
                'news_count', 'avg_confidence'
            ]
            
            return daily_sentiment
            
        except Exception as e:
            logging.error(f"Error analyzing news impact: {str(e)}")
            return pd.DataFrame()
    
    def extract_keywords(self, texts: List[str], top_k: int = 10) -> List[str]:
        """提取关键词"""
        try:
            # 使用TF-IDF提取关键词
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words=['的', '了', '在', '是', '和', '与']  # 简单的停用词
            )
            
            # 转换文本
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # 获取特征名称
            feature_names = vectorizer.get_feature_names_out()
            
            # 计算每个词的平均TF-IDF得分
            avg_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # 获取top_k关键词
            top_indices = avg_tfidf.argsort()[-top_k:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            return keywords
            
        except Exception as e:
            logging.error(f"Error extracting keywords: {str(e)}")
            return [] 