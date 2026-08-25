from pathlib import Path
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler


class Scaler:
    def __init__(self, input_cols):
        self.input_cols = list(input_cols)
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, data: pd.DataFrame):
        missing = [
            col for col in self.input_cols
            if col not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing scaler input columns: {missing}"
            )

        self.scaler.fit(
            data[self.input_cols]
        )

        self.fitted = True
        return self

    def scale(self, data: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError(
                "Scaler must be fitted before scaling data."
            )

        data = data.copy()

        scaled = self.scaler.transform(
            data[self.input_cols]
        )

        for i, col in enumerate(self.input_cols):
            data[f"{col}_scaled"] = scaled[:, i]

        return data

    def unscale(self, data: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError(
                "Scaler must be fitted before unscaling data."
            )

        data = data.copy()

        scaled_cols = [
            f"{col}_scaled"
            for col in self.input_cols
        ]

        missing = [
            col for col in scaled_cols
            if col not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing scaled columns: {missing}"
            )

        unscaled = self.scaler.inverse_transform(
            data[scaled_cols]
        )

        for i, col in enumerate(self.input_cols):
            data[col] = unscaled[:, i]

        return data

    def save(self, path):
        if not self.fitted:
            raise RuntimeError(
                "Cannot save an unfitted scaler."
            )

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        path = Path(path)

        with open(path, "rb") as f:
            scaler = pickle.load(f)

        if not isinstance(scaler, cls):
            raise TypeError(
                f"Loaded object is not a {cls.__name__}."
            )

        return scaler
    

    def get_mean_std(self, cols):
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted.")

        indices = [
            self.input_cols.index(col)
            for col in cols
        ]

        mean = self.scaler.mean_[indices]
        std = self.scaler.scale_[indices]

        return mean, std