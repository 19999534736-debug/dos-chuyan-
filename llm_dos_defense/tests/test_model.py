import numpy as np
import pandas as pd
from src.model import DetectionModel, EnsembleDetectionModel, LightweightDetectionModel


def test_detection_model_training_and_prediction():
    X = pd.DataFrame(np.random.randn(50, 5), columns=[f'feat_{i}' for i in range(5)])
    y = np.random.randint(0, 2, size=50)

    model = DetectionModel(model_type='random_forest', n_estimators=10, max_depth=5)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (50,)
    assert set(preds).issubset({0, 1})

    proba = model.predict_proba(X)
    assert proba.shape == (50, 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_ensemble_detection_model_predict():
    X = pd.DataFrame(np.random.randn(50, 5), columns=[f'feat_{i}' for i in range(5)])
    y = np.random.randint(0, 2, size=50)

    config = {'models': ['random_forest', 'gradient_boosting', 'svm'], 'voting': 'soft'}
    model = EnsembleDetectionModel(config)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (50,)
    assert set(preds).issubset({0, 1})

    proba = model.predict_proba(X)
    assert proba.shape == (50, 2)


def test_lightweight_detection_model():
    detector = LightweightDetectionModel({
        'max_length': 100,
        'max_repetition_ratio': 0.5,
        'min_entropy': 1.0,
        'max_unusual_token_ratio': 0.7,
    })

    normal_text = "What is machine learning?"
    malicious_text = "spam spam spam spam spam spam spam"

    normal_result = detector.detect(normal_text)
    malicious_result = detector.detect(malicious_text)

    assert isinstance(normal_result, tuple)
    assert isinstance(malicious_result, tuple)
    assert normal_result[0] in (True, False)
    assert malicious_result[0] in (True, False)
