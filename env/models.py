"""
env/models.py

All typed Pydantic models used across the ETL environment.
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------

class ActionType(str, Enum):
    IMPUTE_MEAN       = "impute_mean"
    IMPUTE_MODE       = "impute_mode"
    IMPUTE_CONSTANT   = "impute_constant"
    DROP_COLUMN       = "drop_column"
    DROP_ROWS         = "drop_rows"
    CONVERT_TYPE      = "convert_type"
    NORMALIZE         = "normalize"
    REMOVE_DUPLICATES = "remove_duplicates"
    RENAME_COLUMN     = "rename_column"
    SPLIT_COLUMN      = "split_column"
    MERGE_COLUMNS     = "merge_columns"


class DoneReason(str, Enum):
    SUCCESS   = "success"
    MAX_STEPS = "max_steps"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ------------------------------------------------------------------
# Action
# ------------------------------------------------------------------

class Action(BaseModel):
    """
    A single transformation action the agent can take.

    Examples:
        Action(type="impute_mean", column="age")
        Action(type="convert_type", column="price", params={"dtype": "float"})
        Action(type="merge_columns", params={"columns": ["first", "last"], "separator": " ", "new_column": "full_name"})
    """
    type: ActionType
    column: Optional[str] = None          # target column (if applicable)
    params: dict[str, Any] = Field(default_factory=dict)  # extra action-specific params


# ------------------------------------------------------------------
# Observation
# ------------------------------------------------------------------

class ColumnStats(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: list[Any]


class DataQualityMetrics(BaseModel):
    completeness: float   # 1 - missing_rate
    uniqueness: float     # 1 - duplicate_rate
    validity: float       # fraction of correctly typed values
    consistency: float    # schema match score vs expected


class Observation(BaseModel):
    """
    What the agent sees at each step.
    """
    step: int
    shape: tuple[int, int]
    columns: list[ColumnStats]
    quality: DataQualityMetrics
    sample_rows: list[dict[str, Any]]     # first 5 rows as records
    action_history: list[dict[str, Any]]  # past actions + rewards
    task_description: str
    remaining_steps: int


# ------------------------------------------------------------------
# Step Result
# ------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------
# Task Config (loaded from YAML)
# ------------------------------------------------------------------

class GraderWeights(BaseModel):
    schema_match: float  = 0.30
    value_similarity: float = 0.40
    model_performance: float = 0.30


class TaskConfig(BaseModel):
    """
    Parsed from tasks/*.yaml
    """
    name: str
    difficulty: Difficulty
    description: str
    dirty_data_path: str
    clean_data_path: str
    success_threshold: float = 0.90    # quality score to consider episode solved
    expected_schema: dict[str, str]    # col_name -> expected dtype string
    grader_weights: GraderWeights = Field(default_factory=GraderWeights)
    corruption_types: list[str] = Field(default_factory=list)  # informational
    max_steps: int = 30