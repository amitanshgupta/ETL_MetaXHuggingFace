"""
grader/grader.py
Final deterministic episode scorer.
Called by environment.py at episode end.
"""

from grader.metrics import schema_match_score, value_similarity_score
from grader.model_eval import model_performance_score
from env.models import TaskConfig
import pandas as pd


# Target column per task — used for ML eval
TASK_TARGETS = {
    "easy_missing":      ("Survived",   "classification"),
    "medium_schema":     ("SalePrice",  "regression"),
    "hard_integration":  ("order_status", "classification"),
}


class Grader:

    def score(
        self,
        cleaned_df: pd.DataFrame,
        clean_df: pd.DataFrame,
        task_config: TaskConfig,
    ) -> float:
        """
        Final score = weighted combination of:
          - schema_match
          - value_similarity
          - model_performance
        All in [0, 1], final score in [0, 1].
        """
        w = task_config.grader_weights

        s_schema = schema_match_score(cleaned_df, task_config.expected_schema)
        s_value  = value_similarity_score(cleaned_df, clean_df)

        target_col, task_type = TASK_TARGETS.get(task_config.name, (None, None))
        if target_col:
            s_model = model_performance_score(cleaned_df, clean_df, target_col, task_type)
        else:
            s_model = 0.0

        final = (
            w.schema_match       * s_schema +
            w.value_similarity   * s_value  +
            w.model_performance  * s_model
        )

        return round(final, 4)