"""
模型训练器模块

功能：
1. 数据加载和预处理
2. 模型训练循环
3. 超参数管理
4. 检查点保存
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


class Trainer:
    """模型训练器"""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "outputs"):
        self.config = config
        self.output_dir = output_dir
        self.best_model = None
        self.best_score = -np.inf
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self, train_path: str, val_path: str = None, test_path: str = None) -> Tuple:
        """加载数据集"""
        from feature_extraction import FeatureExtractor
        from utils import load_config
        
        logger.info("Loading datasets...")
        
        # 加载训练集
        train_df = pd.read_csv(train_path)
        X_train = self._prepare_features(train_df)
        y_train = train_df['label'].values
        
        logger.info(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"  Positive: {(y_train == 1).sum()}, Negative: {(y_train == 0).sum()}")
        
        result = (X_train, y_train)
        
        # 加载验证集
        if val_path:
            val_df = pd.read_csv(val_path)
            X_val = self._prepare_features(val_df)
            y_val = val_df['label'].values
            logger.info(f"Val set: {X_val.shape[0]} samples")
            logger.info(f"  Positive: {(y_val == 1).sum()}, Negative: {(y_val == 0).sum()}")
            result = result + (X_val, y_val)
        
        # 加载测试集
        if test_path:
            test_df = pd.read_csv(test_path)
            X_test = self._prepare_features(test_df)
            y_test = test_df['label'].values
            logger.info(f"Test set: {X_test.shape[0]} samples")
            logger.info(f"  Positive: {(y_test == 1).sum()}, Negative: {(y_test == 0).sum()}")
            result = result + (X_test, y_test)
        
        return result
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备特征"""
        from feature_extraction import FeatureExtractor
        
        feature_extractor = FeatureExtractor(self.config)
        
        # 提取特征
        features_list = []
        for text in tqdm(df['text'], desc="Extracting features", leave=False):
            features = feature_extractor.extract(text)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # 处理缺失值
        features_df = features_df.fillna(0)
        
        # 移除无穷大
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        return features_df
    
    def train(self, model, X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[np.ndarray] = None):
        """训练模型"""
        logger.info("Starting training...")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 评估
        from evaluator import Evaluator
        evaluator = Evaluator()
        
        # 训练集评估
        y_pred_train = model.predict(X_train)
        y_proba_train = model.predict_proba(X_train)
        
        train_metrics = evaluator.evaluate(y_train, y_pred_train, y_proba_train[:, 1])
        logger.info(f"Train metrics: {train_metrics}")
        
        # 验证集评估
        if X_val is not None and y_val is not None:
            y_pred_val = model.predict(X_val)
            y_proba_val = model.predict_proba(X_val)
            
            val_metrics = evaluator.evaluate(y_val, y_pred_val, y_proba_val[:, 1])
            logger.info(f"Val metrics: {val_metrics}")
            
            # 保存最好的模型
            if val_metrics['f1_score'] > self.best_score:
                self.best_score = val_metrics['f1_score']
                self.best_model = model
                model.save(os.path.join(self.output_dir, 'best_model.pkl'))
                logger.info(f"Saved best model with F1: {self.best_score:.4f}")
        else:
            self.best_model = model
        
        return model


class SimpleTrainingPipeline:
    """简单的训练管道"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run(self, data_dir: str, output_dir: str = "outputs"):
        """运行完整的训练管道"""
        import os
        from model import DetectionModel, EnsembleDetectionModel
        from evaluator import Evaluator
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 数据加载
        self.logger.info("=" * 60)
        self.logger.info("Step 1: Loading data")
        self.logger.info("=" * 60)
        
        trainer = Trainer(self.config, output_dir)
        
        train_path = os.path.join(data_dir, "train.csv")
        val_path = os.path.join(data_dir, "val.csv")
        test_path = os.path.join(data_dir, "test.csv")
        
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_data(
                train_path, val_path, test_path
            )
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return None
        
        # 2. 模型训练
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Step 2: Training model")
        self.logger.info("=" * 60)
        
        if self.config['model']['detector_type'] == 'ensemble':
            model = EnsembleDetectionModel(self.config['model']['ensemble'])
        else:
            model = DetectionModel(
                model_type=self.config['model']['detector_type'],
                **self.config['model'].get(self.config['model']['detector_type'], {})
            )
        
        trainer.train(model, X_train, y_train, X_val, y_val)
        
        # 3. 模型评估
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Step 3: Evaluating model")
        self.logger.info("=" * 60)
        
        evaluator = Evaluator()
        
        # 测试集评估
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)
        
        test_metrics = evaluator.evaluate(y_test, y_pred_test, y_proba_test[:, 1])
        
        self.logger.info("\nTest set results:")
        for metric_name, metric_value in sorted(test_metrics.items()):
            self.logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # 保存评估报告
        report = {
            'config': self.config,
            'test_metrics': test_metrics,
            'model_type': self.config['model']['detector_type'],
        }
        
        report_path = os.path.join(output_dir, "evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"\nEvaluation report saved to {report_path}")
        
        # 4. 特征重要性分析
        if hasattr(model, 'models'):
            self.logger.info("\n" + "=" * 60)
            self.logger.info("Step 4: Feature importance (first model)")
            self.logger.info("=" * 60)
            
            first_model = model.models[0][1] if hasattr(model.models[0], '__len__') else model.models[0]
            if hasattr(first_model, 'feature_importances_'):
                importances = first_model.feature_importances_
                feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f"feature_{i}" for i in range(len(importances))]
                
                feature_importance = sorted(
                    zip(feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                self.logger.info("\nTop 10 important features:")
                for name, importance in feature_importance[:10]:
                    self.logger.info(f"  {name}: {importance:.4f}")
        
        return model


if __name__ == "__main__":
    from utils import load_config
    
    # 加载配置
    config = load_config("configs/config.yaml")
    
    # 运行训练管道
    pipeline = SimpleTrainingPipeline(config)
    model = pipeline.run("data/processed", "outputs")
