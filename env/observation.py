"""
env/observation.py
Builds a typed Observation from the current DataFrame state.
"""

import pandas as pd
from env.models import Observation, ColumnStats, DataQualityMetrics, TaskConfig
from env.utils import compute_validity, compute_consistency, missing_rate, duplicate_rate

class ObservationBuilder:

    def build(self, df: pd.DataFrame, action_history: list, step: int, task_config: TaskConfig) -> Observation:
        return Observation(
            step=step,
            shape=tuple(df.shape),
            columns=self._col_stats(df),
            quality=self._quality_metrics(df, task_config),
            sample_rows=df.head(5).fillna("NULL").to_dict(orient="records"),
            action_history=action_history,
            task_description=task_config.description,
            remaining_steps=task_config.max_steps - step,
        )

    def _col_stats(self, df: pd.DataFrame) -> list[ColumnStats]:
        stats = []
        for col in df.columns:
            missing = int(df[col].isna().sum())
            stats.append(ColumnStats(
                name=col,
                dtype=str(df[col].dtype),
                missing_count=missing,
                missing_pct=round(missing / len(df), 4) if len(df) > 0 else 0.0,
                unique_count=int(df[col].nunique()),
                sample_values=df[col].dropna().head(3).tolist(),
            ))
        return stats

    def _quality_metrics(self, df: pd.DataFrame, task_config: TaskConfig) -> DataQualityMetrics:
        return DataQualityMetrics(
            completeness=round(1.0 - missing_rate(df), 4),
            uniqueness=round(1.0 - duplicate_rate(df), 4),
            validity=compute_validity(df, task_config.expected_schema),
            consistency=compute_consistency(df, task_config.expected_schema),
        )

        return DataQualityMetrics(
            completeness=round(completeness, 4),
            uniqueness=round(uniqueness, 4),
            validity=round(validity, 4),
            consistency=round(consistency, 4),
        )

