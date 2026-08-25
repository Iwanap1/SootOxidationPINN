from pathlib import Path
import pickle

import pandas as pd


class Imputer:
    MATERIAL_COLS = ["sup_calcination_temp", "sup_calcination_time", "Sbet"]
    def __init__(self, fill_config):
        self.fill_config = fill_config.copy()
        self.fill_values = {}
        self.fitted = False

    def fit(self, data: pd.DataFrame):
        missing_cols = [
            col for col in self.fill_config
            if col not in data.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing imputer columns: {missing_cols}"
            )

        for col, strategy in self.fill_config.items():
            source = data

            if col in self.MATERIAL_COLS:
                if "material_id" not in data.columns:
                    raise ValueError(
                        f"Cannot impute material column '{col}' "
                        "because 'material_id' is missing."
                    )

                source = data.drop_duplicates(
                    subset="material_id"
                )

            if strategy == "mean":
                value = source[col].mean()

            elif strategy == "median":
                value = source[col].median()

            elif isinstance(strategy, (int, float)):
                value = float(strategy)

            else:
                raise ValueError(
                    f"Unknown fill strategy for {col}: {strategy}"
                )

            if pd.isna(value):
                raise ValueError(
                    f"Could not determine fill value for {col}"
                )

            self.fill_values[col] = value

        self.fitted = True
        return self

    def transform(self, data: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError(
                "Imputer must be fitted before transforming data."
            )

        data = data.copy()

        for col, value in self.fill_values.items():
            data[col] = data[col].fillna(value)

        return data

    def fit_transform(self, data: pd.DataFrame):
        self.fit(data)
        return self.transform(data)

    def save(self, path):
        if not self.fitted:
            raise RuntimeError(
                "Cannot save an unfitted imputer."
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
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object is not a {cls.__name__}."
            )

        return obj