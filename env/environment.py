"""
env/environment.py

Core ETL OpenEnv Environment.
Orchestrates dataset state, action dispatch, reward computation, and observation building.
"""

import copy
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional

from env.models import Observation, Action, StepResult, TaskConfig, DoneReason
from env.actions import ActionDispatcher
from env.observation import ObservationBuilder
from env.reward import RewardComputer
from grader.grader import Grader


class ETLEnvironment:
    """
    OpenEnv-compliant environment for autonomous data cleaning & ETL.

    Lifecycle:
        env = ETLEnvironment(task_path="tasks/easy_missing.yaml")
        obs = env.reset()
        while not done:
            result = env.step(action)
            obs, reward, done, info = result
    """

    MAX_STEPS = 30  # prevent infinite loops; agent must clean efficiently

    def __init__(self, task_path: str):
        self.task_path = Path(task_path)
        self.task_config: TaskConfig = self._load_task(self.task_path)

        self._dispatcher = ActionDispatcher()
        self._obs_builder = ObservationBuilder()
        self._reward_computer = RewardComputer()
        self._grader = Grader()

        # Runtime state (populated on reset)
        self._dirty_df: Optional[pd.DataFrame] = None   # current working dataset
        self._clean_df: Optional[pd.DataFrame] = None   # ground truth
        self._action_history: list[dict] = []
        self._step_count: int = 0
        self._prev_quality: float = 0.0
        self._done: bool = False

    # ------------------------------------------------------------------
    # OpenEnv Interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset environment to initial dirty dataset. Returns first observation."""
        dirty_path = Path(self.task_config.dirty_data_path)
        clean_path = Path(self.task_config.clean_data_path)

        self._dirty_df = pd.read_csv(dirty_path)
        self._clean_df = pd.read_csv(clean_path)
        self._action_history = []
        self._step_count = 0
        self._done = False

        # Compute initial quality baseline so reward can measure delta
        self._prev_quality = self._reward_computer.compute_quality(
            self._dirty_df, self._clean_df, self.task_config
        )

        return self._obs_builder.build(
            df=self._dirty_df,
            action_history=self._action_history,
            step=self._step_count,
            task_config=self.task_config,
        )

    def step(self, action: Action) -> StepResult:
        """
        Apply action to current dataset.
        Returns (observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping.")

        self._step_count += 1
        info: dict = {"step": self._step_count, "action": action.model_dump()}

        # --- Apply action ---
        try:
            new_df, action_meta = self._dispatcher.dispatch(
                df=self._dirty_df,
                action=action,
            )
            info["action_status"] = "success"
            info.update(action_meta)
        except Exception as e:
            # Invalid action: penalize and return same state
            reward = self._reward_computer.invalid_action_penalty()
            obs = self._obs_builder.build(
                df=self._dirty_df,
                action_history=self._action_history,
                step=self._step_count,
                task_config=self.task_config,
            )
            info["action_status"] = "failed"
            info["error"] = str(e)
            self._action_history.append({"action": action.model_dump(), "reward": reward, "status": "failed"})
            return StepResult(observation=obs, reward=reward, done=False, info=info)

        # --- Compute reward ---
        new_quality = self._reward_computer.compute_quality(new_df, self._clean_df, self.task_config)
        reward = self._reward_computer.compute_reward(
            prev_quality=self._prev_quality,
            new_quality=new_quality,
            action=action,
            df_before=self._dirty_df,
            df_after=new_df,
        )

        # --- Commit state ---
        self._dirty_df = new_df
        self._prev_quality = new_quality
        self._action_history.append({"action": action.model_dump(), "reward": reward, "status": "success"})

        # --- Check termination ---
        done_reason = self._check_done(new_quality)
        self._done = done_reason is not None
        info["done_reason"] = done_reason.value if done_reason else None
        info["quality_score"] = new_quality

        # Final grader score on episode end
        if self._done:
            final_score = self._grader.score(
                cleaned_df=self._dirty_df,
                clean_df=self._clean_df,
                task_config=self.task_config,
            )
            info["final_score"] = final_score

        obs = self._obs_builder.build(
            df=self._dirty_df,
            action_history=self._action_history,
            step=self._step_count,
            task_config=self.task_config,
        )

        return StepResult(observation=obs, reward=reward, done=self._done, info=info)

    def state(self) -> dict:
        """
        Full internal state snapshot.
        Useful for checkpointing, debugging, and API serialization.
        """
        return {
            "task": self.task_config.model_dump(),
            "step": self._step_count,
            "done": self._done,
            "quality_score": self._prev_quality,
            "action_history": self._action_history,
            "dataset_shape": list(self._dirty_df.shape) if self._dirty_df is not None else None,
            "columns": list(self._dirty_df.columns) if self._dirty_df is not None else None,
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _load_task(self, path: Path) -> TaskConfig:
        """Parse YAML task config into typed TaskConfig model."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return TaskConfig(**raw)

    def _check_done(self, quality: float) -> Optional[DoneReason]:
        """Determine if episode should terminate."""
        if quality >= self.task_config.success_threshold:
            return DoneReason.SUCCESS
        if self._step_count >= self.MAX_STEPS:
            return DoneReason.MAX_STEPS
        return None