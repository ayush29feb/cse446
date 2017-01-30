import pandas as pd
import numpy as np

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, :1].values.reshape(-1)

    return (X, y)