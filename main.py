from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os

app = FastAPI()

# Setup templates directory relative to this file
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "../frontend")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ✅ Load model
model = joblib.load(r"C:\Users\mahaj\btc_model.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def predict(
    request: Request,
    days: int = Form(...),
    sma_7: float = Form(...),
    sma_30: float = Form(...),
    volatility: float = Form(...)
):
    input_data = np.array([[days, sma_7, sma_30, volatility]])
    prediction = float(model.predict(input_data)[0])
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": f"${prediction:,.2f}"
    })
