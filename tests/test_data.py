import numpy as np
import pandas as pd

from ecg_mlops.data import LABEL_COL, N_TIMESTEPS, prepare_arrays, validate_raw_dataframe


def _synthetic_dataframe(rows_per_class=6):
    rows = []
    for label in range(5):
        for _ in range(rows_per_class):
            signal = np.linspace(0, 0.95, N_TIMESTEPS) + label * 0.01
            rows.append(np.append(signal, label))
    return pd.DataFrame(rows)


def test_validate_raw_dataframe_accepts_expected_schema():
    df = _synthetic_dataframe()
    validate_raw_dataframe(df)
    assert df.shape[1] == LABEL_COL + 1


def test_prepare_arrays_returns_expected_shapes():
    df_train = _synthetic_dataframe(rows_per_class=8)
    df_test = _synthetic_dataframe(rows_per_class=4)

    arrays = prepare_arrays(df_train, df_test, validation_size=0.25, seed=42)

    assert arrays.X_train.shape[1] == N_TIMESTEPS
    assert arrays.X_val.shape[1] == N_TIMESTEPS
    assert arrays.X_test.shape[1] == N_TIMESTEPS
    assert set(arrays.y_train) == {0, 1, 2, 3, 4}
