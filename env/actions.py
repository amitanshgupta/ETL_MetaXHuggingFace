"""
env/actions.py
Dispatches agent actions onto the working DataFrame.
Returns (new_df, meta) — never mutates in place.
"""

import pandas as pd
from env.models import Action, ActionType


class ActionDispatcher:

    def dispatch(self, df: pd.DataFrame, action: Action) -> tuple[pd.DataFrame, dict]:
        df = df.copy()
        col = action.column
        p = action.params
        meta = {}

        match action.type:

            case ActionType.IMPUTE_MEAN:
                self._require_col(df, col)
                before = df[col].isna().sum()
                df[col] = df[col].fillna(df[col].mean())
                meta["filled"] = int(before - df[col].isna().sum())

            case ActionType.IMPUTE_MODE:
                self._require_col(df, col)
                before = df[col].isna().sum()
                df[col] = df[col].fillna(df[col].mode()[0])
                meta["filled"] = int(before - df[col].isna().sum())

            case ActionType.IMPUTE_CONSTANT:
                self._require_col(df, col)
                value = p.get("value")
                if value is None:
                    raise ValueError("impute_constant requires params.value")
                df[col] = df[col].fillna(value)

            case ActionType.DROP_COLUMN:
                self._require_col(df, col)
                df = df.drop(columns=[col])

            case ActionType.DROP_ROWS:
                # drops rows where `col` is null, or all-null rows if col is None
                before = len(df)
                df = df.dropna(subset=[col] if col else None)
                meta["dropped"] = before - len(df)

            case ActionType.CONVERT_TYPE:
                self._require_col(df, col)
                dtype = p.get("dtype")
                if not dtype:
                    raise ValueError("convert_type requires params.dtype")
                df[col] = df[col].astype(dtype, errors="ignore")  # soft fail per cell

            case ActionType.NORMALIZE:
                self._require_col(df, col)
                col_min, col_max = df[col].min(), df[col].max()
                if col_max == col_min:
                    raise ValueError(f"Column '{col}' has zero variance, cannot normalize.")
                df[col] = (df[col] - col_min) / (col_max - col_min)

            case ActionType.REMOVE_DUPLICATES:
                before = len(df)
                subset = p.get("subset")  # optional list of cols
                df = df.drop_duplicates(subset=subset)
                meta["dropped"] = before - len(df)

            case ActionType.RENAME_COLUMN:
                self._require_col(df, col)
                new_name = p.get("new_name")
                if not new_name:
                    raise ValueError("rename_column requires params.new_name")
                df = df.rename(columns={col: new_name})

            case ActionType.SPLIT_COLUMN:
                self._require_col(df, col)
                sep = p.get("separator", " ")
                new_cols = p.get("new_columns")  # e.g. ["first_name", "last_name"]
                if not new_cols:
                    raise ValueError("split_column requires params.new_columns")
                split = df[col].str.split(sep, expand=True)
                for i, name in enumerate(new_cols):
                    df[name] = split[i] if i < split.shape[1] else None
                df = df.drop(columns=[col])

            case ActionType.MERGE_COLUMNS:
                cols = p.get("columns")
                sep = p.get("separator", " ")
                new_col = p.get("new_column")
                if not cols or not new_col:
                    raise ValueError("merge_columns requires params.columns and params.new_column")
                df[new_col] = df[cols].astype(str).agg(sep.join, axis=1)
                df = df.drop(columns=cols)

            case _:
                raise ValueError(f"Unknown action type: {action.type}")

        return df, meta

    def _require_col(self, df: pd.DataFrame, col: str | None):
        if not col:
            raise ValueError("This action requires a column name.")
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")