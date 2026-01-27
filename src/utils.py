# Files containing utility functions
import pandas as pd
from typing import Union

def text_to_numeric(data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    if isinstance(data, pd.DataFrame):
        serie = data.iloc[:, 0]
    else:
        serie = data

    cleaned = pd.to_numeric(serie.astype(str).str.extract(r'(\d+)')[0])
    return cleaned.fillna(0)
