import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


def test_linear_regression_predictions_are_numeric_and_expected():
    features = pd.DataFrame({"hours_studied": [1, 2, 3, 4]})
    targets = pd.Series([3, 5, 7, 9])

    model = LinearRegression()
    model.fit(features, targets)

    predictions = model.predict(pd.DataFrame({"hours_studied": [5, 6]}))

    assert predictions.shape == (2,)
    assert np.issubdtype(predictions.dtype, np.number)
    assert predictions == pytest.approx([11, 13])
