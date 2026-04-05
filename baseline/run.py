"""
baseline/run.py
Runs heuristic agent on all 3 tasks, prints reproducible baseline scores.
"""

from env.environment import ETLEnvironment
from baseline.agent import HeuristicAgent

TASKS = [
    "tasks/easy_missing.yaml",
    "tasks/medium_schema.yaml",
    "tasks/hard_integration.yaml",
]


def run_episode(task_path: str) -> dict:
    env = ETLEnvironment(task_path=task_path)
    agent = HeuristicAgent()

    obs = env.reset()
    done = False
    total_reward = 0.0
    result = None

    while not done:
        action = agent.act(obs)
        if action is None:
            break
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward
        done = result.done

    # Force final score regardless of how episode ended
    from grader.grader import Grader
    final_score = Grader().score(
        cleaned_df=env._dirty_df,
        clean_df=env._clean_df,
        task_config=env.task_config,
    )

    state = env.state()
    return {
        "task":         state["task"]["name"],
        "difficulty":   state["task"]["difficulty"],
        "steps":        state["step"],
        "total_reward": round(total_reward, 4),
        "quality":      state["quality_score"],
        "final_score":  final_score,
    }


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  ETL OpenEnv — Baseline Agent Results")
    print(f"{'='*55}")
    for task_path in TASKS:
        r = run_episode(task_path)
        print(f"\nTask:         {r['task']} ({r['difficulty']})")
        print(f"Steps taken:  {r['steps']}")
        print(f"Total reward: {r['total_reward']:+.4f}")
        print(f"Quality:      {r['quality']:.4f}")
        print(f"Final score:  {r['final_score']}")
    print(f"\n{'='*55}\n")