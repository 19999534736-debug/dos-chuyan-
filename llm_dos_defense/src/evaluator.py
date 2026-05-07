"""
模型评估器模块

功能：
1. 计算各种评估指标
2. 生成混淆矩阵
3. ROC-AUC分析
4. 性能分析
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, auc
)
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Evaluator:
    """模型评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率（可选）
        
        Returns:
            指标字典
        """
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # 基于概率的指标
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except Exception as e:
                metrics['roc_auc'] = 0.0
                logger.warning(f"Failed to compute ROC-AUC: {e}")
            
            # PR-AUC
            try:
                precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
                metrics['pr_auc'] = auc(recall_vals, precision_vals)
            except Exception as e:
                metrics['pr_auc'] = 0.0
                logger.warning(f"Failed to compute PR-AUC: {e}")
        
        # 混淆矩阵的其他指标
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        self.metrics = metrics
        return metrics
    
    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """打印分类报告"""
        print("\n" + "=" * 60)
        print("Classification Report")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Malicious']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                       output_path: Optional[str] = None):
        """绘制ROC曲线"""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"ROC curve saved to {output_path}")
            
            plt.show()
        except Exception as e:
            logger.warning(f"Failed to plot ROC curve: {e}")
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    output_path: Optional[str] = None):
        """绘制精确率-召回率曲线"""
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = auc(recall_vals, precision_vals)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="best")
            plt.grid(alpha=0.3)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"PR curve saved to {output_path}")
            
            plt.show()
        except Exception as e:
            logger.warning(f"Failed to plot PR curve: {e}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             output_path: Optional[str] = None):
        """绘制混淆矩阵"""
        try:
            import seaborn as sns
            
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Normal', 'Malicious'],
                       yticklabels=['Normal', 'Malicious'])
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {output_path}")
            
            plt.show()
        except Exception as e:
            logger.warning(f"Failed to plot confusion matrix: {e}")


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.latency_measurements = []
        self.memory_measurements = []
    
    def measure_inference_latency(self, model, X: pd.DataFrame,
                                 num_runs: int = 100) -> Dict[str, float]:
        """测量推理延迟"""
        import time
        
        latencies = []
        
        # 预热
        _ = model.predict(X.iloc[:1])
        
        # 测量
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.predict(X.iloc[:1])
            latency = (time.time() - start_time) * 1000  # 转换为毫秒
            latencies.append(latency)
        
        latencies = np.array(latencies)
        
        stats = {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
        }
        
        self.latency_measurements = latencies
        return stats
    
    def measure_throughput(self, model, X: pd.DataFrame,
                          batch_size: int = 32) -> Dict[str, float]:
        """测量吞吐量"""
        import time
        
        # 准备批次
        num_batches = len(X) // batch_size
        total_samples = num_batches * batch_size
        
        # 预热
        _ = model.predict(X.iloc[:batch_size])
        
        # 测量
        start_time = time.time()
        for i in range(num_batches):
            batch = X.iloc[i*batch_size:(i+1)*batch_size]
            _ = model.predict(batch)
        total_time = time.time() - start_time
        
        throughput = total_samples / total_time
        
        return {
            'throughput_samples_per_second': throughput,
            'total_time_seconds': total_time,
            'total_samples': total_samples,
        }
    
    def measure_model_size(self, model_path: str) -> Dict[str, float]:
        """测量模型大小"""
        import os
        
        size_bytes = os.path.getsize(model_path)
        
        return {
            'model_size_bytes': size_bytes,
            'model_size_mb': size_bytes / (1024 ** 2),
            'model_size_kb': size_bytes / 1024,
        }


class ThresholdAnalyzer:
    """阈值分析器"""
    
    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                              metric: str = 'f1') -> Tuple[float, float]:
        """
        找到最优的分类阈值
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            metric: 优化指标 ('f1', 'roc_auc', 'precision', 'recall')
        
        Returns:
            (最优阈值, 最优指标值)
        """
        best_threshold = 0.5
        best_value = 0.0
        
        for threshold in np.arange(0.1, 0.99, 0.01):
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                value = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                value = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                value = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'roc_auc':
                try:
                    value = roc_auc_score(y_true, y_pred)
                except:
                    value = 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if value > best_value:
                best_value = value
                best_threshold = threshold
        
        return best_threshold, best_value
    
    @staticmethod
    def analyze_threshold_trade_offs(y_true: np.ndarray, y_proba: np.ndarray):
        """分析不同阈值的权衡关系"""
        thresholds = np.arange(0.0, 1.01, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # 测试评估器
    np.random.seed(42)
    
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.85, 0.9, 0.6, 0.2])
    
    # 评估
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_true, y_pred, y_proba)
    
    print("Metrics:")
    for metric_name, metric_value in sorted(metrics.items()):
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # 阈值分析
    threshold_analyzer = ThresholdAnalyzer()
    optimal_threshold, optimal_f1 = threshold_analyzer.find_optimal_threshold(y_true, y_proba, 'f1')
    print(f"\nOptimal threshold: {optimal_threshold:.2f} (F1: {optimal_f1:.4f})")
