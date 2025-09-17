# resampling_methods.py
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
import numpy as np

def random_oversample(X, y, random_state=42):
    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res

def smote_resample(X, y, random_state=42):
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def adasyn_resample(X, y, random_state=42):
    ad = ADASYN(random_state=random_state)
    X_res, y_res = ad.fit_resample(X, y)
    return X_res, y_res
