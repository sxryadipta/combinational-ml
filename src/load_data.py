import pandas as pd

def load_dataset(path):

    df = pd.read_csv(path)

    X = df[['VL','HL','gates','inputs','outputs']]
    y = df['power']

    return X.values, y.values