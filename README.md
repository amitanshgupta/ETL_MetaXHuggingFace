---
title: ETL OpenEnv
emoji: 🧹
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# 🧹 ETL OpenEnv — Autonomous Data Cleaning Environment

A benchmark environment where an AI agent performs step-by-step data cleaning
and transformation (ETL) on messy datasets to produce clean, model-ready data.

Built for the **OpenEnv Hackathon**.

🔗 **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/YOUR_USERNAME/etl-openenv)  
📖 **API Docs:** `/docs`  
🖥️ **UI:** `/`

---

## 🧠 Core Idea

This is **NOT** a static pipeline.
It is a sequential decision-making environment where an agent:
- Observes a dirty dataset
- Chooses cleaning actions one step at a time
- Receives a reward signal based on data quality improvement
- Tries to match the ground truth clean dataset

---

## 🗂️ Project Structure
etl-openenv/
├── env/
│   ├── environment.py       # Core ETLEnvironment (reset, step, state)
│   ├── models.py            # Pydantic models (Action, Observation, TaskConfig...)
│   ├── actions.py           # ActionDispatcher — transforms DataFrame
│   ├── observation.py       # ObservationBuilder — agent's view each step
│   ├── reward.py            # RewardComputer — dense reward shaping
│   └── utils.py             # Shared helpers
├── data/
│   ├── raw/                 # Original CSVs (not modified)
│   ├── clean/               # Ground truth (generated)
│   ├── dirty/               # Corrupted input (generated)
│   └── generators/
│       ├── dataset_loader.py
│       └── corruption.py
├── tasks/
│   ├── easy_missing.yaml
│   ├── medium_schema.yaml
│   └── hard_integration.yaml
├── grader/
│   ├── grader.py
│   ├── metrics.py
│   └── model_eval.py
├── baseline/
│   ├── agent.py
│   └── run.py
├── api/
│   ├── app.py
│   └── index.html
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md

---

## ⚙️ How It Works

### Environment Lifecycle
```python
from env.environment import ETLEnvironment
from env.models import Action, ActionType

env = ETLEnvironment(task_path="tasks/easy_missing.yaml")
obs = env.reset()

while not done:
    action = Action(type=ActionType.IMPUTE_MEAN, column="Age")
    result = env.step(action)
    obs, reward, done, info = result.observation, result.reward, result.done, result.info
```

### Action Space

| Action | Description |
|---|---|
| `impute_mean` | Fill nulls with column mean |
| `impute_mode` | Fill nulls with most frequent value |
| `impute_constant` | Fill nulls with a fixed value |
| `drop_column` | Remove a column entirely |
| `drop_rows` | Drop rows where a column is null |
| `convert_type` | Cast column to target dtype |
| `normalize` | Min-max normalize a numeric column |
| `remove_duplicates` | Drop duplicate rows |
| `rename_column` | Rename a column |
| `split_column` | Split one column into multiple |
| `merge_columns` | Merge multiple columns into one |

### Observation (what the agent sees)
- Dataset shape and sample rows
- Per-column stats (dtype, missing %, unique count)
- Data quality metrics (completeness, uniqueness, validity, consistency)
- Full action history with rewards
- Steps remaining

### Reward Signal
Reward = ΔQuality + Penalties

| Component | Value |
|---|---|
| Quality improvement | +ΔQ |
| Redundant action | -0.05 |
| Destructive drop (>20% rows) | -0.15 |
| Invalid action | -0.10 |

Quality score:
Q = 0.35×completeness + 0.25×validity + 0.20×consistency + 0.20×uniqueness

### Tasks

| Task | Dataset | Difficulty | Corruptions | Threshold |
|---|---|---|---|---|
| `easy_missing` | Titanic | 🟢 Easy | Missing values, duplicates | 0.90 |
| `medium_schema` | House Prices | 🟡 Medium | Type mismatches, noise | 0.88 |
| `hard_integration` | Olist E-Commerce | 🔴 Hard | Multi-source, conflicts | 0.85 |

### Grader

Final score = weighted combination of:
- **Schema match** — correct dtypes vs expected schema
- **Value similarity** — numeric MAE + categorical exact match vs ground truth
- **Model performance** — downstream ML accuracy on cleaned data

All scores deterministic, normalized to [0, 1].

### Baseline Scores (Heuristic Agent)

| Task | Quality | Final Score |
|---|---|---|
| easy_missing | 0.8958 | 0.8606 |
| medium_schema | 0.9096 | 0.8746 |
| hard_integration | 0.8056 | 0.4858 |

---

## 🚀 Setup

### 1. Install
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/etl-openenv
cd etl-openenv
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Download Raw Datasets
Place in `data/raw/`:

| File | Source |
|---|---|
| `titanic.csv` | [Kaggle — Titanic](https://www.kaggle.com/datasets/yasserh/titanic-dataset) |
| `house_prices_train.csv` | [Kaggle — House Prices](https://www.kaggle.com/datasets/lespin/house-prices-dataset) — use `train.csv` |
| `olist_orders_dataset.csv` | [Kaggle — Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |
| `olist_customers_dataset.csv` | same Olist download |
| `olist_order_items_dataset.csv` | same Olist download |

### 3. Generate Datasets
```bash
python -m data.generators.dataset_loader
python -m data.generators.corruption
```

### 4. Run Baseline
```bash
python -m baseline.run
```

### 5. Start API
```bash
python -m api.app
# UI at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 6. Docker
```bash
docker build -t etl-openenv .
docker run -p 8000:8000 etl-openenv
```

---

## 🐳 Deployment (Hugging Face Spaces)

This space uses the **Docker SDK**.
The app is served via FastAPI on port `7860` (HF default).

Push to HF:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/etl-openenv
git push hf main
```

---

## 🧩 Key Design Decisions

- **No in-place mutation** — every action copies the DataFrame, safe for reward delta
- **Deterministic corruption** — `seed=42`, fully reproducible
- **Dense rewards** — agent always gets a signal, no sparse reward problem
- **Typed everything** — Pydantic models across all boundaries
- **YAML-driven tasks** — new task = new YAML, zero code changes

---

## 📦 Requirements
fastapi
uvicorn
pydantic
pandas
numpy
scikit-learn
pyyaml
jinja2

