from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pydantic import BaseModel
from pathlib import Path

BASE = Path(__file__).parent

app = FastAPI(title="Student Placement Predictor")
templates = Jinja2Templates(directory=str(BASE / "templates"))


class StudentScores(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)


def load_scaler():
    df = pd.read_csv(BASE / "student.csv")
    df["placement_status"] = df["placement_status"].apply(lambda x: 1 if x == "Placed" else 0)
    X = df[["study_hours", "sleep_hours", "attendance", "exam_score"]].values
    y = df["placement_status"].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    sc.fit(X_train)
    return sc


scaler = load_scaler()

model = StudentScores()
model.load_state_dict(torch.load(BASE / "models/student_scores.pth", map_location="cpu"))
model.eval()


class PredictionInput(BaseModel):
    study_hours: float
    sleep_hours: float
    attendance: float
    exam_score: float


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(data: PredictionInput):
    X = np.array([[data.study_hours, data.sleep_hours, data.attendance, data.exam_score]])
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.inference_mode():
        logit = model(X_tensor)
        prob = torch.sigmoid(logit).item()

    return {
        "placed": prob >= 0.5,
        "prediction": "Placed" if prob >= 0.5 else "Not Placed",
        "probability": round(prob * 100, 1),
    }
