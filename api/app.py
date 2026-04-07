"""
api/app.py
FastAPI wrapper exposing the ETL OpenEnv interface over HTTP.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from env.environment import ETLEnvironment
from env.models import Action, StepResult, Observation

app = FastAPI(
    title="ETL OpenEnv API",
    description="Autonomous Data Cleaning & ETL Environment",
    version="1.0.0",
)

# Single global environment instance per session
# For multi-user: replace with session-based dict
_env: Optional[ETLEnvironment] = None


# ------------------------------------------------------------------
# Request Models
# ------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_path: str  # e.g. "tasks/easy_missing.yaml"


class StepRequest(BaseModel):
    action: Action


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    global _env
    try:
        _env = ETLEnvironment(task_path=req.task_path)
        obs = _env.reset()
        return obs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    try:
        result = _env.step(req.action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return _env.state()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/actions")
def list_actions():
    """Returns all valid action types for agent reference."""
    from env.models import ActionType
    return {"actions": [a.value for a in ActionType]}

from pathlib import Path

@app.get("/", response_class=HTMLResponse)
def ui():
    file_path = Path(__file__).parent / "index.html"
    if not file_path.exists():
        raise HTTPException(status_code=500, detail="index.html not found")

    return file_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)