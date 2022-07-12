import pandas as pd


def read_file(path: str) -> pd.DataFrame:
    return pd.read_json(path)

