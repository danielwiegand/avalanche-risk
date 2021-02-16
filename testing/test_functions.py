import sys

sys.path.append("/home/daniel/Schreibtisch/Projekte/avalanche-risk")

import pandas as pd
import numpy as np
from model.functions_model import preprocess_X_values, get_shifted_features
import pytest

@pytest.fixture 
def df(): 
    df = pd.DataFrame([["a", "1"], ["b", "2"], ["c", "3"], ["d", "4"]], index = [1, 2, 3, 4], columns = ["A", "B"])
    return df

def test_get_shifted_features1(df):
    result = get_shifted_features(df, min_shift = 1, max_shift = 2)
    assert len(result) == 2
    assert result.columns.tolist() == ['A-1', 'B-1', 'A-2', 'B-2']
    assert result.loc[3, "A-1"] == "b"
    