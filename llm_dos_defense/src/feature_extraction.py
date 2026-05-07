"""
特征提取模块

功能：
1. 文本统计特征提取
2. 语义特征提取
3. 模式特征提取
4. 特征标准化
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
from collections import Counter
import re
from utils import setup_logger, calculate_entropy, calculate_repetition_ratio, calculate_unique_ratio

logger = setup_logger(__name__)

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("Transformers not installed, semantic features will be disabled")


class StatisticalFeatureExtractor:
    """统计特征提取器"""
    
    def __init__(self):
        self.feature_names = [
            'input_length',
            'word_count',
            'unique_word_ratio',
            'avg_word_length',
            'special_char_ratio',
            'digit_ratio',
            'uppercase_ratio',
            'punctuation_ratio',
            'input_entropy',
            'line_count',
            'avg_line_length',
        ]
    
    def extract(self, text: str) -> Dict[str, float]:
        """提取统计特征"""
        features = {}
        
        if not text:
            return {name: 0.0 for name in self.feature_names}
        
        # 基础长度特征
        features['input_length'] = len(text)
        
        # 词级别特征
        words = text.split()
        features['word_count'] = len(words)
        features['unique_word_ratio'] = calculate_unique_ratio(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0.0
        
        # 字符级别特征
        special_chars = len([c for c in text if not c.isalnum() and c != ' '])
        features['special_char_ratio'] = special_chars / len(text) if text else 0.0
        
        digits = len([c for c in text if c.isdigit()])
        features['digit_ratio'] = digits / len(text) if text else 0.0
        
        uppercase = len([c for c in text if c.isupper()])
        features['uppercase_ratio'] = uppercase / len(text) if text else 0.0
        
        punctuation = len([c for c in text if c in '.,!?;:\'"()-'])
        features['punctuation_ratio'] = punctuation / len(text) if text else 0.0
        
        # 熵特征
        features['input_entropy'] = calculate_entropy(text)
        
        # 行级别特征
        lines = text.split('\n')
        features['line_count'] = len(lines)
        features['avg_line_length'] = np.mean([len(line) for line in lines]) if lines else 0.0
        
        return features
    
    def extract_batch(self, texts: List[str]) -> pd.DataFrame:
        """批量提取特征"""
        features_list = []
        for text in texts:
            features_list.append(self.extract(text))
        
        return pd.DataFrame(features_list)


class PatternFeatureExtractor:
    """模式特征提取器"""
    
    def __init__(self):
        self.feature_names = [
            'repetition_ratio',
            'unusual_token_ratio',
            'bracket_mismatch_ratio',
            'repeated_sequence_ratio',
            'anomaly_token_ratio',
        ]
    
    def extract(self, text: str) -> Dict[str, float]:
        """提取模式特征"""
        features = {}
        
        if not text:
            return {name: 0.0 for name in self.feature_names}
        
        # 重复率
        features['repetition_ratio'] = calculate_repetition_ratio(text, min_repeat_length=3)
        
        # 异常token比例（检查看起来不规则的token）
        words = text.split()
        unusual_count = 0
        for word in words:
            # 检查是否是gibberish
            if self._is_gibberish(word):
                unusual_count += 1
        features['unusual_token_ratio'] = unusual_count / len(words) if words else 0.0
        
        # 括号不匹配率
        open_brackets = text.count('(') + text.count('[') + text.count('{')
        close_brackets = text.count(')') + text.count(']') + text.count('}')
        features['bracket_mismatch_ratio'] = abs(open_brackets - close_brackets) / max(open_brackets + close_brackets, 1)
        
        # 重复序列比例
        features['repeated_sequence_ratio'] = self._calculate_repeated_sequence_ratio(text)
        
        # 异常token检测（基于token频率）
        features['anomaly_token_ratio'] = self._calculate_anomaly_ratio(words)
        
        return features
    
    def _is_gibberish(self, word: str, vowel_ratio_threshold: float = 0.3) -> bool:
        """检查单词是否为gibberish"""
        if len(word) < 3:
            return False
        
        # 检查元音比例
        vowels = 'aeiouAEIOU'
        vowel_count = sum(1 for c in word if c in vowels)
        vowel_ratio = vowel_count / len(word)
        
        # gibberish通常元音少
        if vowel_ratio < vowel_ratio_threshold:
            return True
        
        # 检查是否含有太多重复字符
        if any(word.count(c) >= len(word) * 0.5 for c in set(word)):
            return True
        
        return False
    
    def _calculate_repeated_sequence_ratio(self, text: str, min_seq_length: int = 3) -> float:
        """计算重复序列比例"""
        total_repeated = 0
        words = text.split()
        
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if words[i] == words[j]:
                    # 找到重复序列
                    seq_len = 1
                    k = 1
                    while i + k < len(words) and j + k < len(words) and words[i+k] == words[j+k]:
                        seq_len += 1
                        k += 1
                    
                    if seq_len >= min_seq_length:
                        total_repeated += seq_len
        
        return min(total_repeated / len(text) if text else 0.0, 1.0)
    
    def _calculate_anomaly_ratio(self, words: List[str], threshold: float = 0.7) -> float:
        """计算异常token比例"""
        if not words:
            return 0.0
        
        # 统计单词频率
        word_freq = Counter(words)
        freq_values = list(word_freq.values())
        
        if not freq_values:
            return 0.0
        
        # 高频词阈值
        freq_threshold = np.percentile(freq_values, 50)
        
        # 统计低频词
        low_freq_count = sum(1 for freq in freq_values if freq < freq_threshold * threshold)
        
        return low_freq_count / len(word_freq) if word_freq else 0.0
    
    def extract_batch(self, texts: List[str]) -> pd.DataFrame:
        """批量提取特征"""
        features_list = []
        for text in texts:
            features_list.append(self.extract(text))
        
        return pd.DataFrame(features_list)


class SemanticFeatureExtractor:
    """语义特征提取器（使用预训练模型）"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.feature_names = [
            'semantic_diversity',
            'embedding_norm',
            'embedding_mean',
            'embedding_std',
        ]
        
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(device)
                self.model.eval()
                logger.info(f"Loaded semantic model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
                self.tokenizer = None
                self.model = None
        else:
            self.tokenizer = None
            self.model = None
    
    def extract(self, text: str) -> Dict[str, float]:
        """提取语义特征"""
        features = {}
        
        if not self.tokenizer or not self.model:
            # 如果没有模型，返回默认值
            return {name: 0.0 for name in self.feature_names}
        
        try:
            # 分词
            inputs = self.tokenizer(
                text[:512],  # 限制长度
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # 获取embedding
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state
            
            # 计算特征
            # 语义多样性：基于embedding的距离
            if embeddings.shape[0] > 1:
                pairwise_dist = torch.cdist(embeddings, embeddings)
                features['semantic_diversity'] = float(pairwise_dist.mean().cpu().numpy())
            else:
                features['semantic_diversity'] = 0.0
            
            # embedding统计
            pooled = embeddings.mean(dim=1)[0].cpu().numpy()
            features['embedding_norm'] = float(np.linalg.norm(pooled))
            features['embedding_mean'] = float(pooled.mean())
            features['embedding_std'] = float(pooled.std())
            
        except Exception as e:
            logger.debug(f"Failed to extract semantic features: {e}")
            features = {name: 0.0 for name in self.feature_names}
        
        return features
    
    def extract_batch(self, texts: List[str], batch_size: int = 32) -> pd.DataFrame:
        """批量提取特征"""
        features_list = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                features_list.append(self.extract(text))
        
        return pd.DataFrame(features_list)


class FeatureExtractor:
    """综合特征提取器"""
    
    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        self.config = config
        self.device = device
        
        self.statistical_extractor = StatisticalFeatureExtractor()
        self.pattern_extractor = PatternFeatureExtractor()
        
        if config['features']['use_semantic_features']:
            self.semantic_extractor = SemanticFeatureExtractor(device=device)
        else:
            self.semantic_extractor = None
        
        # 特征名称
        self.feature_names = []
        if config['features']['use_statistical_features']:
            self.feature_names.extend(self.statistical_extractor.feature_names)
        if config['features']['use_pattern_features']:
            self.feature_names.extend(self.pattern_extractor.feature_names)
        if config['features']['use_semantic_features'] and self.semantic_extractor:
            self.feature_names.extend(self.semantic_extractor.feature_names)
        
        logger.info(f"Feature extractor initialized with {len(self.feature_names)} features")
    
    def extract(self, text: str) -> Dict[str, float]:
        """提取所有特征"""
        features = {}
        
        if self.config['features']['use_statistical_features']:
            features.update(self.statistical_extractor.extract(text))
        
        if self.config['features']['use_pattern_features']:
            features.update(self.pattern_extractor.extract(text))
        
        if self.config['features']['use_semantic_features'] and self.semantic_extractor:
            features.update(self.semantic_extractor.extract(text))
        
        return features
    
    def extract_batch(self, texts: List[str]) -> pd.DataFrame:
        """批量提取特征"""
        logger.info(f"Extracting features for {len(texts)} samples...")
        
        features_list = []
        
        # 统计特征
        if self.config['features']['use_statistical_features']:
            stat_df = self.statistical_extractor.extract_batch(texts)
            features_list.append(stat_df)
        
        # 模式特征
        if self.config['features']['use_pattern_features']:
            pattern_df = self.pattern_extractor.extract_batch(texts)
            features_list.append(pattern_df)
        
        # 语义特征
        if self.config['features']['use_semantic_features'] and self.semantic_extractor:
            semantic_df = self.semantic_extractor.extract_batch(texts)
            features_list.append(semantic_df)
        
        # 合并特征
        if features_list:
            df = pd.concat(features_list, axis=1)
        else:
            df = pd.DataFrame()
        
        logger.info(f"Extracted {df.shape[1]} features")
        return df


if __name__ == "__main__":
    from utils import load_config
    
    # 测试特征提取
    config = load_config("configs/config.yaml")
    
    # 测试文本
    normal_text = "What is machine learning?"
    malicious_text = "aaaaaa bbbbbb cccccc aaaa bbbb cccc aaaa bbbb cccc " * 20
    
    print("Normal text:")
    print(f"  Length: {len(normal_text)}")
    print(f"  Repetition ratio: {calculate_repetition_ratio(normal_text):.4f}")
    print(f"  Entropy: {calculate_entropy(normal_text):.4f}")
    
    print("\nMalicious text:")
    print(f"  Length: {len(malicious_text)}")
    print(f"  Repetition ratio: {calculate_repetition_ratio(malicious_text):.4f}")
    print(f"  Entropy: {calculate_entropy(malicious_text):.4f}")
    
    # 特征提取
    extractor = FeatureExtractor(config, device="cpu")
    
    print("\nExtracting features...")
    normal_features = extractor.extract(normal_text)
    malicious_features = extractor.extract(malicious_text)
    
    print("\nNormal text features:")
    for name, value in normal_features.items():
        print(f"  {name}: {value:.4f}")
    
    print("\nMalicious text features:")
    for name, value in malicious_features.items():
        print(f"  {name}: {value:.4f}")
