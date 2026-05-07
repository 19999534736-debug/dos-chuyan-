"""数据集管理模块。

该模块用于构建、保存和加载训练/验证/测试数据集。
"""

import os
from typing import Tuple, Dict
import pandas as pd
from src.data_preparation import DatasetBuilder
from src.utils import load_config


def build_and_save_datasets(config_path: str = "configs/config.yaml") -> Tuple[str, str, str]:
    """根据配置构建并保存数据集。"""
    config = load_config(config_path)
    builder = DatasetBuilder(config)
    df, _ = builder.build_dataset()
    train_df, val_df, test_df = builder.split_dataset(df)

    processed_dir = config['data']['processed_data_path']
    os.makedirs(processed_dir, exist_ok=True)

    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')
    test_path = os.path.join(processed_dir, 'test.csv')

    builder.save_dataset(train_df, train_path)
    builder.save_dataset(val_df, val_path)
    builder.save_dataset(test_df, test_path)

    return train_path, val_path, test_path


def load_dataset(split: str, config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """加载指定数据集切分。"""
    config = load_config(config_path)
    processed_dir = config['data']['processed_data_path']
    split_map = {
        'train': 'train.csv',
        'val': 'val.csv',
        'test': 'test.csv',
    }
    if split not in split_map:
        raise ValueError(f"Unsupported split: {split}")

    path = os.path.join(processed_dir, split_map[split])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return pd.read_csv(path)
