"""
env/reward.py
Computes dense, shaped rewards based on quality delta between steps.
"""

import pandas as pd
from env.models import Action, ActionType, TaskConfig
from env.utils import compute_validity, compute_consistency, missing_rate, duplicate_rate

class RewardComputer:

    # Weights for quality score components
    W_COMPLETENESS = 0.35
    W_UNIQUENESS   = 0.20
    W_VALIDITY     = 0.25
    W_CONSISTENCY  = 0.20

    # Penalty constants
    PENALTY_INVALID_ACTION   = -0.10
    PENALTY_REDUNDANT_ACTION = -0.05
    PENALTY_DESTRUCTIVE      = -0.15  # e.g. dropping too many rows

    DESTRUCTIVE_DROP_THRESHOLD = 0.20  # if action drops >20% rows, penalize

    def compute_quality(self, df: pd.DataFrame, clean_df: pd.DataFrame, task_config: TaskConfig) -> float:
        completeness = 1.0 - missing_rate(df)
        uniqueness   = 1.0 - duplicate_rate(df)
        validity     = compute_validity(df, task_config.expected_schema)
        consistency  = compute_consistency(df, task_config.expected_schema)

        return round(
            self.W_COMPLETENESS * completeness +
            self.W_UNIQUENESS   * uniqueness   +
            self.W_VALIDITY     * validity     +
            self.W_CONSISTENCY  * consistency,
            4
        )
    

    def compute_reward(
        self,
        prev_quality: float,
        new_quality: float,
        action: Action,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
    ) -> float:
        """
        Reward = quality delta + penalties.
        Dense: agent always gets a signal, positive or negative.
        """
        delta = new_quality - prev_quality

        penalty = 0.0

        # Penalize destructive drops
        if action.type == ActionType.DROP_ROWS:
            drop_frac = (len(df_before) - len(df_after)) / len(df_before)
            if drop_frac > self.DESTRUCTIVE_DROP_THRESHOLD:
                penalty += self.PENALTY_DESTRUCTIVE

        # Penalize redundant actions (quality didn't change at all)
        if abs(delta) < 1e-6:
            penalty += self.PENALTY_REDUNDANT_ACTION

        return round(delta + penalty, 4)

    def invalid_action_penalty(self) -> float:
        return self.PENALTY_INVALID_ACTION

    # ------------------------------------------------------------------
    # Internal — mirrors ObservationBuilder but used for reward signal
    # ------------------------------------------------------------------