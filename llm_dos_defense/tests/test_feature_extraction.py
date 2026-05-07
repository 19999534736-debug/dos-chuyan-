import numpy as np
from src.feature_extraction import StatisticalFeatureExtractor, PatternFeatureExtractor, FeatureExtractor
from src.utils import calculate_entropy, calculate_repetition_ratio


def test_calculate_entropy():
    assert calculate_entropy("") == 0.0
    assert calculate_entropy("aaaa") == 0.0
    assert calculate_entropy("abcd") > 0.0


def test_calculate_repetition_ratio():
    assert calculate_repetition_ratio("") == 0.0
    assert calculate_repetition_ratio("aaaaaa") > 0.0
    assert calculate_repetition_ratio("abcdef") == 0.0


def test_statistical_feature_extraction():
    extractor = StatisticalFeatureExtractor()
    features = extractor.extract("Hello World! 123")
    assert features['input_length'] == len("Hello World! 123")
    assert features['word_count'] == 3
    assert features['special_char_ratio'] >= 0.0
    assert features['input_entropy'] >= 0.0


def test_pattern_feature_extraction():
    extractor = PatternFeatureExtractor()
    text = "repeat repeat repeat weirdword 123"
    features = extractor.extract(text)
    assert 0.0 <= features['repetition_ratio'] <= 1.0
    assert 0.0 <= features['unusual_token_ratio'] <= 1.0
    assert 0.0 <= features['bracket_mismatch_ratio'] <= 1.0


def test_feature_extractor_combines_features():
    config = {
        'features': {
            'use_statistical_features': True,
            'use_pattern_features': True,
            'use_semantic_features': False,
        }
    }
    extractor = FeatureExtractor(config)
    features = extractor.extract("Hello world!")
    assert 'input_length' in features
    assert 'repetition_ratio' in features
    assert features['input_length'] > 0
