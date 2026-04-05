"""
baseline/agent.py
Simple heuristic agent — no learning, rule-based action selection.
Used to generate reproducible baseline scores.
"""

from env.models import Action, ActionType, Observation


class HeuristicAgent:

    def act(self, obs: Observation) -> Action | None:
        """
        Priority order:
        1. Remove duplicates (always first)
        2. Impute numeric nulls with mean
        3. Impute categorical nulls with mode
        4. Convert obviously mistyped columns
        5. Return None if nothing to do (episode should end)
        """
        # Step 1 — duplicates (only once, check history)
        past_actions = [a["action"]["type"] for a in obs.action_history]
        if "remove_duplicates" not in past_actions:
            return Action(type=ActionType.REMOVE_DUPLICATES)

        # Step 2 & 3 — handle missing columns
        for col in obs.columns:
            if col.missing_count == 0:
                continue
            if col.dtype in ("float64", "int64", "int32"):
                return Action(type=ActionType.IMPUTE_MEAN, column=col.name)
            else:
                return Action(type=ActionType.IMPUTE_MODE, column=col.name)

        # Step 4 — fix type mismatches
        for col in obs.columns:
            if col.dtype == "object":
                # try to cast to numeric if sample values look numeric
                samples = col.sample_values
                numeric_looking = all(
                    str(v).replace(".", "").replace("-", "").isdigit()
                    for v in samples if str(v) not in ("nan", "NULL", "N/A", "")
                )
                if numeric_looking and samples:
                    return Action(type=ActionType.CONVERT_TYPE, column=col.name, params={"dtype": "float"})

# Step 5 — fallback: drop columns with >50% missing
        for col in obs.columns:
            if col.missing_pct > 0.50:
                return Action(type=ActionType.DROP_COLUMN, column=col.name)
        return None  # nothing left to do