import os
import pandas as pd
from src.data_preparation import DatasetBuilder
from src.utils import load_config
from src.model import DetectionModel
from src.trainer import Trainer


def test_dataset_builder_and_trainer(tmp_path):
    config = {
        'data': {
            'normal_samples': 20,
            'malicious_samples': 20,
            'train_ratio': 0.5,
            'val_ratio': 0.25,
            'random_seed': 42,
            'processed_data_path': str(tmp_path / 'processed'),
        },
        'features': {
            'use_statistical_features': True,
            'use_pattern_features': True,
            'use_semantic_features': False,
        },
        'model': {
            'detector_type': 'random_forest',
            'random_forest': {
                'n_estimators': 10,
                'max_depth': 5,
                'min_samples_split': 2,
                'n_jobs': 1,
            }
        }
    }

    builder = DatasetBuilder(config)
    df, _ = builder.build_dataset()
    train_df, val_df, test_df = builder.split_dataset(df)

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert set(train_df['label'].unique()) <= {0, 1}

    trainer = Trainer(config, output_dir=str(tmp_path / 'output'))
    X_train = trainer._prepare_features(train_df)
    y_train = train_df['label'].values

    model = DetectionModel(model_type='random_forest', **config['model']['random_forest'])
    model = trainer.train(model, X_train, y_train)

    preds = model.predict(X_train)
    assert preds.shape == y_train.shape
