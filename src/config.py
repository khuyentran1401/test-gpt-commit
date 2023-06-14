"""
create Pydantic models
"""
from typing import List

from pydantic import BaseModel, validator


def must_be_non_negative(v: float) -> float:
    """Check if the v is non-negative

    Parameters
    ----------
    v : float
        value

    Returns
    -------
    float
        v

    Raises
    ------
    ValueError
        Raises error when v is negative
    """
    if v < 0:
        raise ValueError(f"{v} must be non-negative")
    return v


class Location(BaseModel):
    """Specify the locations of inputs and outputs"""

    data_raw: str = "data/raw/iris.csv"
    data_process: str = "data/processed/xy.pkl"
    data_final: str = "data/final/predictions.pkl"
    model: str = "models/tree.pkl"
    input_notebook: str = "notebooks/analyze_results.ipynb"
    output_notebook: str = "notebooks/results.ipynb"


class ProcessConfig(BaseModel):
    """Specify the parameters of the `process` flow"""

    drop_columns: List[str] = ["Id"]
    label: str = "Species"
    test_size: float = 0.3

    _validated_test_size = validator("test_size", allow_reuse=True)(
        must_be_non_negative
    )


class ModelParams(BaseModel):
    """Specify the parameters of the `train` flow"""

    max_depth: List[int] = [1, 2, 3, 4, 5]
    min_samples_split: List[int] = [2, 5, 10, 20]
    min_samples_leaf: List[int] = [1, 2, 5, 10]

    _validated_fields = validator("*", allow_reuse=True, each_item=True)(
        must_be_non_negative
    )
