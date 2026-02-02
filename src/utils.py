# Files containing utility functions
import pandas as pd
import numpy as np
from typing import Union, List
from sklearn.pipeline import Pipeline
from sklearn.base import clone


def text_to_numeric(data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Extracts the first sequence of digits from a string and converts it to a numeric value.
    @param {Union[pd.Series, pd.DataFrame]} data - The input data containing strings to parse.
    @returns {pd.DataFrame} - A single-column DataFrame containing the extracted numbers.
    """
    # Ensure we are working with a Series even if a DataFrame is passed
    if isinstance(data, pd.DataFrame):
        serie = data.iloc[:, 0]
    else:
        serie = data

    # Extract digits using regex, convert to numeric type, and handle missing values
    cleaned = pd.to_numeric(serie.astype(str).str.extract(r"(\d+)")[0])
    return cleaned.fillna(0).to_frame()


def get_upper_matrix(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a correlation matrix to keep only the upper triangle.
    @param {pd.DataFrame} correlation_matrix - The full correlation matrix.
    @returns {pd.DataFrame} - The matrix with NaN values in the lower triangle and diagonal.
    """
    # np.triu generates a mask for the upper triangle; k=1 excludes the diagonal
    return correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )


def get_correlated(
    half_correlation_matrix: pd.DataFrame, threshold: float = 0.9
) -> List[str]:
    """
    Identifies columns that have a correlation coefficient higher than the specified threshold.
    @param {pd.DataFrame} half_correlation_matrix - An upper or lower triangle correlation matrix.
    @param {float} threshold - The absolute correlation threshold (default 0.9).
    @returns {List[str]} - A list of column names exceeding the threshold.
    """
    # Identify columns where any value (absolute) is greater than the threshold
    return [
        column
        for column in half_correlation_matrix.columns
        if any(abs(half_correlation_matrix[column]) > threshold)
    ]


def create_pipeline(preprocessor, model_name, model, extra_steps=None) -> Pipeline:
    steps = [("preprocessor", clone(preprocessor))]

    if extra_steps:
        steps.extend(extra_steps)

    steps.append((model_name, model))

    return Pipeline(steps)


def remove_redundancy(dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return dataframe.drop(columns=columns)
