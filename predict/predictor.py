#!/usr/bin/env python3
import os
import numpy as np

from libs.models.dt import RandomForestEcgModel
from libs.features import feature_extractor5
from libs.loading import loader
from libs.preprocessing import categorizer, normalizer
from libs.utils.common import set_seed


extractor = feature_extractor5

def get_saved_model():
    model = RandomForestEcgModel()
    model.restore()
    return model

def classify(record, data_dir):
    x = loader.load_data_from_file(record, data_dir)
    x = normalizer.normalize_ecg(x)
    x = extractor.features_for_row(x)

    clf = get_saved_model()

    x = np.array(x).reshape(1, -1)
    return categorizer.get_original_label(clf.predict(x)[0])

def main_classify_single():
    label = classify('matlab_file', 'predict')
    os.remove('predict/matlab_file.mat')
    return label

def start_predict():
    set_seed(42)
    return main_classify_single()
