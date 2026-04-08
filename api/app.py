"""
api/app.py
FastAPI wrapper exposing the ETL OpenEnv interface over HTTP.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import uvicorn
import traceback

from env.environment import ETLEnvironment
from env.models import Action, StepResult, Observation


app = FastAPI(
    title="ETL OpenEnv API",
    description="Autonomous Data Cleaning & ETL Environment",
    version="1.0.0",
)

# Allow frontend access (important for deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance per session
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
        return _env.reset()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    try:
        return _env.step(req.action)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    try:
        return _env.state()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/actions")
def list_actions():
    """Returns all valid action types for agent reference."""
    from env.models import ActionType
    return {"actions": [a.value for a in ActionType]}


# ------------------------------------------------------------------
# UI Route (Robust + Deployment Safe)
# ------------------------------------------------------------------

@app.get("/")
def ui():
    file_path = Path(__file__).parent / "index.html"

    if not file_path.exists():
        raise HTTPException(status_code=500, detail="index.html not found")

    return FileResponse(file_path)


# ------------------------------------------------------------------
# Local Run
# ------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)