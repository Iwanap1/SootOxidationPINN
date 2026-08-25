import numpy as np
import pandas as pd


class DataSplitter:
    def __init__(
        self,
        train_fraction=0.7,
        eval_fraction=0.15,
        test_fraction=0.15,
        split_on="experiment",
        experiment_id_col="_id_experiment",
        material_id_col="material_id",
        random_state=1,
    ):
        self.train_fraction = train_fraction
        self.eval_fraction = eval_fraction
        self.test_fraction = test_fraction
        self.split_on = split_on
        self.experiment_id_col = experiment_id_col
        self.material_id_col = material_id_col
        self.random_state = random_state

        self._validate_args()

    def _validate_args(self):
        total = (
            self.train_fraction
            + self.eval_fraction
            + self.test_fraction
        )

        if not np.isclose(total, 1.0):
            raise ValueError(
                "train_fraction + eval_fraction + "
                "test_fraction must equal 1."
            )

        if self.split_on not in {"experiment", "material"}:
            raise ValueError(
                "split_on must be either 'experiment' or 'material'."
            )

        for name, value in {
            "train_fraction": self.train_fraction,
            "eval_fraction": self.eval_fraction,
            "test_fraction": self.test_fraction,
        }.items():
            if value < 0 or value > 1:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

    def split(self, df: pd.DataFrame):
        df = df.copy()

        if self.split_on == "experiment":
            split_col = self.experiment_id_col
        else:
            split_col = self.material_id_col

        if split_col not in df.columns:
            raise ValueError(
                f"Split column '{split_col}' not found in dataframe."
            )

        if df[split_col].isna().any():
            raise ValueError(
                f"Split column '{split_col}' contains NaN values."
            )

        unique_ids = df[split_col].unique()

        rng = np.random.default_rng(self.random_state)
        unique_ids = rng.permutation(unique_ids)

        n_total = len(unique_ids)

        n_train = int(
            np.floor(n_total * self.train_fraction)
        )

        n_eval = int(
            np.floor(n_total * self.eval_fraction)
        )

        train_ids = unique_ids[:n_train]

        eval_ids = unique_ids[
            n_train:n_train + n_eval
        ]

        test_ids = unique_ids[
            n_train + n_eval:
        ]

        train_df = df[
            df[split_col].isin(train_ids)
        ].copy()

        eval_df = df[
            df[split_col].isin(eval_ids)
        ].copy()

        test_df = df[
            df[split_col].isin(test_ids)
        ].copy()

        self._check_no_leakage(
            train_df,
            eval_df,
            test_df,
            split_col,
        )

        return train_df, eval_df, test_df

    def _check_no_leakage(
        self,
        train_df,
        eval_df,
        test_df,
        split_col,
    ):
        train_ids = set(train_df[split_col].unique())
        eval_ids = set(eval_df[split_col].unique())
        test_ids = set(test_df[split_col].unique())

        if train_ids & eval_ids:
            raise RuntimeError(
                "ID leakage between train and eval sets."
            )

        if train_ids & test_ids:
            raise RuntimeError(
                "ID leakage between train and test sets."
            )

        if eval_ids & test_ids:
            raise RuntimeError(
                "ID leakage between eval and test sets."
            )

    def print_summary(
        self,
        train_df,
        eval_df,
        test_df,
    ):
        if self.split_on == "experiment":
            split_col = self.experiment_id_col
        else:
            split_col = self.material_id_col

        total_rows = (
            len(train_df)
            + len(eval_df)
            + len(test_df)
        )

        total_ids = (
            train_df[split_col].nunique()
            + eval_df[split_col].nunique()
            + test_df[split_col].nunique()
        )

        print(
            f"Split on: {self.split_on} "
            f"({split_col})"
        )

        for name, split_df in [
            ("Train", train_df),
            ("Eval", eval_df),
            ("Test", test_df),
        ]:
            n_rows = len(split_df)
            n_ids = split_df[split_col].nunique()

            print(
                f"{name:5s}: "
                f"{n_ids:4d} IDs "
                f"({n_ids / total_ids:.1%}), "
                f"{n_rows:5d} rows "
                f"({n_rows / total_rows:.1%})"
            )