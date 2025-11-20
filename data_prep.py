import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def generate_synthetic_dataset(n=1000, seed=42):
    """Creates a multivariate synthetic time-series dataset."""
    np.random.seed(seed)
    t = np.arange(0, n)
    data = pd.DataFrame({
        "value1": np.sin(0.02 * t) + np.random.normal(0, 0.1, len(t)),
        "value2": np.cos(0.02 * t) + np.random.normal(0, 0.1, len(t)),
        "value3": 0.5 * np.sin(0.04 * t) + np.random.normal(0, 0.1, len(t))
    })
    return data

def scale_dataset(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler

def create_sequences(data, window=50, target_col=0):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window][target_col])  # predicting first feature by default
    return np.array(X), np.array(y)
