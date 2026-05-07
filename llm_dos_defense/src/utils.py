"""
工具函数和辅助类
"""

import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import pickle

# 配置日志
def setup_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """设置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 配置加载
def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """保存配置到YAML文件"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


# 数据持久化
def save_pickle(obj: Any, path: str):
    """保存对象到pickle文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """从pickle文件加载对象"""
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_json(data: Any, path: str, indent: int = 2):
    """保存数据到JSON文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Any:
    """从JSON文件加载数据"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# 统计函数
def calculate_entropy(text: str) -> float:
    """计算文本的信息熵"""
    if not text:
        return 0.0
    
    # 计算字符频率
    char_freq = {}
    for char in text:
        char_freq[char] = char_freq.get(char, 0) + 1
    
    # 计算熵
    entropy = 0.0
    text_len = len(text)
    for freq in char_freq.values():
        p = freq / text_len
        entropy -= p * np.log2(p)
    
    return entropy


def calculate_repetition_ratio(text: str, min_repeat_length: int = 3) -> float:
    """计算文本重复率"""
    if len(text) < min_repeat_length:
        return 0.0
    
    total_repeat_chars = 0
    
    # 检查连续重复
    i = 0
    while i < len(text):
        j = i
        while j < len(text) and text[j] == text[i]:
            j += 1
        
        repeat_len = j - i
        if repeat_len >= min_repeat_length:
            total_repeat_chars += repeat_len
        
        i = j
    
    return min(total_repeat_chars / len(text), 1.0)


def calculate_unique_ratio(tokens: List[str]) -> float:
    """计算token唯一性比例"""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def update(self, **kwargs):
        """更新指标"""
        self.metrics.update(kwargs)
    
    def record(self):
        """记录当前指标到历史"""
        self.history.append(self.metrics.copy())
    
    def get_best(self, metric_name: str) -> Tuple[float, int]:
        """获取最好的指标值和对应的epoch"""
        if not self.history:
            return None, -1
        
        values = [h.get(metric_name, float('-inf')) for h in self.history]
        best_idx = np.argmax(values)
        return values[best_idx], best_idx
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame(self.history)
    
    def to_dict(self) -> Dict[str, List]:
        """转换为字典"""
        result = {}
        if self.history:
            keys = self.history[0].keys()
            for key in keys:
                result[key] = [h[key] for h in self.history]
        return result


class ConfusionMatrix:
    """混淆矩阵"""
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes))
    
    def update(self, pred: int, target: int):
        """更新混淆矩阵"""
        self.matrix[target, pred] += 1
    
    def update_batch(self, preds: np.ndarray, targets: np.ndarray):
        """批量更新"""
        for pred, target in zip(preds, targets):
            self.update(int(pred), int(target))
    
    def get_metrics(self) -> Dict[str, float]:
        """计算各种指标"""
        tp = self.matrix[1, 1]
        fp = self.matrix[0, 1]
        fn = self.matrix[1, 0]
        tn = self.matrix[0, 0]
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
    
    def __str__(self) -> str:
        return f"Confusion Matrix:\n{self.matrix}\n" + \
               "\n".join([f"{k}: {v:.4f}" for k, v in self.get_metrics().items()])


def create_output_dir(base_dir: str, prefix: str = "exp") -> str:
    """创建带时间戳的输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        base_dir, 
        f"{prefix}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def print_section(title: str, width: int = 80):
    """打印分割线"""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width + "\n")


if __name__ == "__main__":
    # 测试
    logger = setup_logger("test", level="INFO")
    logger.info("Logger initialized")
    
    # 测试熵计算
    text1 = "aaaaa"
    text2 = "hello"
    print(f"Entropy of '{text1}': {calculate_entropy(text1):.4f}")
    print(f"Entropy of '{text2}': {calculate_entropy(text2):.4f}")
    
    # 测试重复率
    text3 = "aaaaabbbccdd"
    print(f"Repetition ratio: {calculate_repetition_ratio(text3):.4f}")
