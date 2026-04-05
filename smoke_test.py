"""
smoke_test.py
Runs a single episode on the easy task with hardcoded actions.
Verifies the full reset → step → done pipeline.
"""

from env.environment import ETLEnvironment
from env.models import Action, ActionType

env = ETLEnvironment(task_path="tasks/easy_missing.yaml")

# --- Reset ---
obs = env.reset()
print(f"Reset OK | shape: {obs.shape} | quality: {obs.quality}")

# --- Step through some actions ---
actions = [
    Action(type=ActionType.REMOVE_DUPLICATES),
    Action(type=ActionType.IMPUTE_MEAN, column="Age"),
    Action(type=ActionType.IMPUTE_MODE, column="Embarked"),
    Action(type=ActionType.IMPUTE_CONSTANT, column="Cabin", params={"value": "Unknown"}),
    Action(type=ActionType.IMPUTE_MEAN, column="Fare"),
]

for action in actions:
    result = env.step(action)
    print(f"Action: {action.type.value:<20} | reward: {result.reward:+.4f} | quality: {result.info['quality_score']:.4f} | done: {result.done}")
    if result.done:
        print(f"Final score: {result.info.get('final_score')}")
        break

# --- State snapshot ---
print("\n--- State Snapshot ---")
state = env.state()
print(f"Step: {state['step']} | Quality: {state['quality_score']} | Done: {state['done']}")
print(f"Shape: {state['dataset_shape']} | Columns: {len(state['columns'])}")