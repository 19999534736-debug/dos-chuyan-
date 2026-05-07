"""
模型定义模块

支持多种模型类型：
- 轻量级分类器（RandomForest, XGBoost等）
- 深度学习模型（LSTM, CNN等）
- 集成学习
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Optional, List
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import json

logger = setup_logger = logging.getLogger(__name__)


class DetectionModel:
    """基础检测模型类"""
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._create_model(**kwargs)
    
    def _create_model(self, **kwargs):
        """创建模型"""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                min_samples_split=kwargs.get("min_samples_split", 5),
                n_jobs=kwargs.get("n_jobs", -1),
                random_state=42,
                verbose=1
            )
        
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                max_depth=kwargs.get("max_depth", 5),
                random_state=42,
                verbose=1
            )
        
        elif self.model_type == "svm":
            self.model = SVC(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 1.0),
                gamma=kwargs.get("gamma", "scale"),
                probability=True,
                verbose=1
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # 创建特征缩放器
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """训练模型"""
        # 保存特征名
        self.feature_names = X.columns.tolist()
        
        # 特征缩放
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_scaled, y)
        
        logger.info("Model training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # SVM的decision_function
            if hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(X_scaled)
                # 转换为概率
                proba = 1 / (1 + np.exp(-decision))
                return np.column_stack([1 - proba, proba])
            else:
                # 直接返回预测
                pred = self.model.predict(X_scaled)
                return np.column_stack([1 - pred, pred])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not support feature importance")
            return {}
        
        importances = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importances))
        
        # 按重要性排序
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, path: str):
        """保存模型"""
        model_data = {
            'model_type': self.model_type,
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DetectionModel':
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {path}")
        return instance


class EnsembleDetectionModel:
    """集成检测模型"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = []
        self.scaler = None
        self.feature_names = None
        self.voting = config.get('voting', 'soft')
        self.ensemble = None
        self._create_ensemble(config)
    
    def _create_ensemble(self, config: Dict[str, Any]):
        """创建集成模型"""
        model_configs = config.get('models', ['random_forest', 'gradient_boosting', 'svm'])
        estimators = []
        
        for model_type in model_configs:
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    n_jobs=-1,
                    random_state=42
                )
            elif model_type == 'gradient_boosting':
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            elif model_type == 'svm':
                model = SVC(
                    kernel='rbf',
                    C=1.0,
                    probability=True,
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.models.append((model_type, model))
            estimators.append((model_type, model))
        
        self.scaler = StandardScaler()
        self.ensemble = VotingClassifier(estimators=estimators, voting=self.voting, n_jobs=-1)
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """训练集成模型"""
        self.feature_names = X.columns.tolist()
        
        # 特征缩放
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练集成模型
        logger.info(f"Training ensemble voting classifier with {len(self.models)} base models...")
        self.ensemble.fit(X_scaled, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        X_scaled = self.scaler.transform(X)
        return self.ensemble.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        X_scaled = self.scaler.transform(X)
        if hasattr(self.ensemble, 'predict_proba'):
            return self.ensemble.predict_proba(X_scaled)
        else:
            pred = self.ensemble.predict(X_scaled)
            return np.column_stack([1 - pred, pred])
    
    def save(self, path: str):
        """保存模型"""
        model_data = {
            'ensemble': self.ensemble,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'voting': self.voting,
            'models': self.models,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Ensemble model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleDetectionModel':
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls.__new__(cls)
        instance.models = model_data.get('models', [])
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.voting = model_data.get('voting', 'soft')
        instance.ensemble = model_data.get('ensemble', None)
        instance.config = {
            'models': [name for name, _ in instance.models],
            'voting': instance.voting,
        }
        
        logger.info(f"Ensemble model loaded from {path}")
        return instance


class LightweightDetectionModel:
    """轻量级快速检测模型
    
    基于启发式规则，用于第一层防御
    """
    
    def __init__(self, threshold_config: Dict[str, float]):
        """
        Args:
            threshold_config: 阈值配置
                - max_length: 最大允许输入长度
                - min_length: 最小输入长度
                - max_repetition_ratio: 最大重复率
                - max_unusual_token_ratio: 最大异常token比例
                - max_entropy: 最大熵（低熵可能表示重复）
        """
        self.threshold_config = {
            'max_length': 500,
            'min_length': 5,
            'max_repetition_ratio': 0.6,
            'max_unusual_token_ratio': 0.7,
            'min_entropy': 1.0,  # 过低的熵表示重复
        }
        self.threshold_config.update(threshold_config)
    
    def detect(self, text: str) -> Tuple[bool, float]:
        """
        快速检测文本是否为恶意
        
        Returns:
            (is_malicious, confidence_score)
        """
        if not text:
            return False, 0.0
        
        from src.utils import calculate_entropy, calculate_repetition_ratio
        
        score = 0.0
        
        # 规则1: 长度异常
        if len(text) > self.threshold_config['max_length']:
            score += 0.3
        
        # 规则2: 重复率过高
        rep_ratio = calculate_repetition_ratio(text, min_repeat_length=3)
        if rep_ratio > self.threshold_config['max_repetition_ratio']:
            score += 0.3
        
        # 规则3: 熵过低（表示重复）
        entropy = calculate_entropy(text)
        if entropy < self.threshold_config['min_entropy']:
            score += 0.2
        
        # 规则4: 特殊字符过多
        special_chars = sum(1 for c in text if not c.isalnum() and c != ' ')
        special_ratio = special_chars / len(text) if text else 0.0
        if special_ratio > 0.5:
            score += 0.2
        
        is_malicious = score > 0.5
        return is_malicious, score


if __name__ == "__main__":
    # 测试模型
    from src.feature_extraction import FeatureExtractor
    from src.utils import load_config
    import os
    
    config = load_config("configs/config.yaml")
    
    # 生成示例数据
    X = pd.DataFrame(np.random.randn(100, 10))
    y = np.random.randint(0, 2, 100)
    
    # 测试单个模型
    print("Testing single model...")
    model = DetectionModel(model_type="random_forest")
    model.fit(X, y)
    
    predictions = model.predict(X)
    print(f"Predictions: {predictions[:10]}")
    
    # 测试集成模型
    print("\nTesting ensemble model...")
    ensemble_config = config['model']['ensemble']
    ensemble_model = EnsembleDetectionModel(ensemble_config)
    ensemble_model.fit(X, y)
    
    ensemble_predictions = ensemble_model.predict(X)
    print(f"Ensemble predictions: {ensemble_predictions[:10]}")
    
    # 测试轻量级模型
    print("\nTesting lightweight model...")
    lightweight_model = LightweightDetectionModel({})
    test_texts = [
        "What is machine learning?",
        "aaaa bbbb cccc aaaa bbbb cccc " * 10,
    ]
    
    for text in test_texts:
        is_malicious, score = lightweight_model.detect(text)
        print(f"Text: '{text[:50]}...'")
        print(f"  Malicious: {is_malicious}, Score: {score:.4f}")
