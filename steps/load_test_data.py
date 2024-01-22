import pandas as pd


def load_test_data():
    data = pd.read_csv("data/processed_data.csv")
    df = data.sample(1200)
    return df