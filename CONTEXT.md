```
You are an expert AI systems architect and backend engineer helping build a hackathon project.

---

## PROJECT OVERVIEW

**Name:** ETL OpenEnv — Autonomous Data Cleaning & ETL Environment
**Goal:** A benchmark environment where an AI agent performs step-by-step data cleaning and transformation (ETL) on messy datasets through sequential decision-making.
**Built for:** OpenEnv Hackathon

The environment exposes three core methods:
- `reset()` — loads dirty dataset, returns first observation
- `step(action)` — applies transformation, returns (observation, reward, done, info)
- `state()` — returns full internal state snapshot

---

## COMPLETE FILE STRUCTURE

```
etl-openenv/
├── env/
│   ├── environment.py       ✅ DONE
│   ├── models.py            ✅ DONE
│   ├── actions.py           ✅ DONE
│   ├── observation.py       ✅ DONE
│   ├── reward.py            ✅ DONE
│   └── utils.py             ✅ DONE
├── data/
│   ├── raw/                 ✅ DONE (CSVs downloaded)
│   ├── clean/               ✅ DONE (generated)
│   ├── dirty/               ✅ DONE (generated)
│   └── generators/
│       ├── dataset_loader.py ✅ DONE
│       └── corruption.py     ✅ DONE
├── tasks/
│   ├── easy_missing.yaml    ✅ DONE
│   ├── medium_schema.yaml   ✅ DONE
│   └── hard_integration.yaml ✅ DONE
├── grader/
│   ├── grader.py            ✅ DONE
│   ├── metrics.py           ✅ DONE
│   └── model_eval.py        ✅ DONE
├── baseline/
│   ├── agent.py             ✅ DONE
│   └── run.py               ✅ DONE
├── api/
│   └── app.py               ✅ DONE
├── openenv.yaml             ✅ DONE
├── requirements.txt         ✅ DONE
├── Dockerfile               ⏳ NOT DONE
└── README.md                ✅ DONE (needs final update after Docker)
```

---

## WHAT EACH FILE DOES

### env/environment.py
Core orchestrator. Holds dataset state, dispatches actions, computes rewards, checks termination.
- `reset()` loads dirty CSV, computes initial quality baseline
- `step(action)` dispatches → reward delta → termination check → returns StepResult
- `state()` returns full snapshot for API/debugging
- Episode ends when `quality >= success_threshold` OR `steps >= MAX_STEPS (30)`

### env/models.py
All Pydantic typed models:
- `Action` — type + optional column + params dict
- `ActionType` — enum of 11 action types
- `Observation` — what agent sees (shape, column stats, quality metrics, sample rows, history)
- `StepResult` — (observation, reward, done, info)
- `TaskConfig` — parsed from YAML (paths, schema, thresholds, grader weights)
- `DataQualityMetrics` — completeness, uniqueness, validity, consistency
- `DoneReason` — SUCCESS or MAX_STEPS

### env/actions.py
`ActionDispatcher.dispatch(df, action)` — applies transformation, returns (new_df, meta).
Always copies df before mutating. Supported actions:
impute_mean, impute_mode, impute_constant, drop_column, drop_rows,
convert_type, normalize, remove_duplicates, rename_column, split_column, merge_columns

### env/observation.py
`ObservationBuilder.build()` — constructs typed Observation from current DataFrame.
Computes per-column stats and quality metrics using utils.py helpers.

### env/reward.py
`RewardComputer` — dense reward shaping.
- Quality = 0.35*completeness + 0.20*uniqueness + 0.25*validity + 0.20*consistency
- Reward = ΔQuality + penalties
- Penalties: redundant action (-0.05), destructive drop (-0.15), invalid action (-0.10)

### env/utils.py
Shared helpers used across reward.py and observation.py:
compute_validity, compute_consistency, missing_rate, duplicate_rate

### data/generators/dataset_loader.py
Loads raw CSVs from data/raw/, does minimal cleaning, saves ground truth to data/clean/.
Run once: `python -m data.generators.dataset_loader`

### data/generators/corruption.py
Takes clean CSVs, injects deterministic corruption (seed=42), saves to data/dirty/.
Corruption types: missing values, type mismatches, duplicates, noise.
Run once: `python -m data.generators.corruption`

### tasks/*.yaml
YAML task configs loaded by ETLEnvironment. Each defines:
- dirty_data_path, clean_data_path
- expected_schema (col -> dtype)
- success_threshold
- grader_weights (schema_match, value_similarity, model_performance)
- corruption_types, max_steps

### grader/metrics.py
Individual scoring metrics (all return float in [0,1]):
- schema_match_score — dtype match vs expected schema
- value_similarity_score — numeric MAE + categorical exact match vs ground truth
- completeness_score, duplicate_score

### grader/model_eval.py
Trains RandomForest on cleaned data, scores against ground truth performance.
Normalized: cleaned_score / baseline_score. Returns float in [0,1].

### grader/grader.py
Final episode scorer. Combines schema_match + value_similarity + model_performance
using per-task weights from TaskConfig. Deterministic, score in [0,1].

### baseline/agent.py
`HeuristicAgent.act(obs)` — rule-based, no learning.
Priority: remove_duplicates → impute_mean/mode → convert_type → drop high-missing columns

### baseline/run.py
Runs heuristic agent on all 3 tasks, prints reproducible baseline scores.
Run: `python -m baseline.run`

Verified baseline scores:
- easy_missing:     0.8606
- medium_schema:    0.8746
- hard_integration: 0.4858

### api/app.py
FastAPI wrapper. Endpoints:
- POST /reset — initializes environment with task_path
- POST /step — applies action, returns StepResult
- GET  /state — returns full state snapshot
- GET  /health — health check
- GET  /actions — lists all valid action types

Run: `python -m api.app`
Interactive docs: http://localhost:8000/docs

---

## DATASETS

| File | Location | Source |
|---|---|---|
| titanic.csv | data/raw/ | Kaggle — yasserh/titanic-dataset |
| house_prices_train.csv | data/raw/ | Kaggle — lespin/house-prices-dataset |
| olist_orders_dataset.csv | data/raw/ | Kaggle — olistbr/brazilian-ecommerce |
| olist_customers_dataset.csv | data/raw/ | same |
| olist_order_items_dataset.csv | data/raw/ | same |

---

## VERIFIED WORKING

- Full reset → step → done pipeline smoke tested
- Rewards are dense and signed correctly
- Redundant action penalty fires correctly
- Grader scores all 3 tasks
- Baseline agent runs reproducibly
- FastAPI server starts and serves endpoints

---

## WHAT IS REMAINING

### 1. Dockerfile ⏳
Containerize the FastAPI app for deployment on Hugging Face Spaces.
Should:
- Use python:3.11-slim base
- Copy all project files
- Install requirements.txt
- Expose port 8000
- Run uvicorn api.app:app

### 2. README.md final update
Add Docker build/run instructions once Dockerfile is done.

---

## DEPENDENCIES

```
fastapi
uvicorn
pydantic
pandas
numpy
scikit-learn
pyyaml
```

Install: `pip install -r requirements.txt`

---

## SETUP FROM SCRATCH

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets into data/raw/ (see DATASETS section above)

# 3. Generate clean ground truth
python -m data.generators.dataset_loader

# 4. Generate dirty datasets
python -m data.generators.corruption

# 5. Run baseline to verify everything works
python -m baseline.run

# 6. Start API
python -m api.app
```

---

## KEY DESIGN DECISIONS TO KEEP IN MIND

- No in-place DataFrame mutation — always copy before modifying
- seed=42 everywhere — fully deterministic and reproducible
- Pydantic models for all inputs/outputs — no raw dicts crossing boundaries
- YAML-driven tasks — new task = new YAML file, zero code changes
- Dense rewards — agent always gets a signal, no sparse reward problem
- Grader only runs at episode end — reward signal during episode is quality delta only

---

Your job is to help finish the remaining Dockerfile, and do a final README update.
Follow the same patterns already established in the codebase.
Do not restructure or refactor anything that is already working.
```
