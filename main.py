import uvicorn
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse
from ModelName import ModelName
import predict


class Prostate(BaseModel):
    PSA: float = Field(..., ge=0, title="PSA at TRUS", description="PSA is a blood test that measures the amount of prostate-specific antigen (PSA) in your blood.")
    DRE: bool = Field(..., title="DRE at TRUS", description="Digital rectal examination (DRE) is a clinical examination of the prostate gland.")
    TRUS_volume: float = Field(..., ge=0, title="TRUS volume", description="TRUS Volume is prostate volume measured by TRUS, an ultrasound technique that is used to view a man's prostate and surrounding tissues")
    TRUS_Lesion: bool = Field(..., title="TRUS Lesion", description="Any abnormal lesion in the prostate gland which is visualised upon TRUS.")


# host_url = "https://machine-learning-calculator.herokuapp.com/"
host_url = "http://idrps.shealth.info/ml/"
# host_url = "http://localhost:8000/"

tags_metadata = [
    {
        "name": "Root",
        "description": "Docs of the API server",
        "externalDocs": {
            "description": "Another docs",
            "url": host_url + "redoc",
        },
    },
    {
        "name": "Prostate cancer",
        "description": "Prostate cancer prediction",
    },
    {
        "name": "Model",
        "description": "Machine learning models",
    },
]

app = FastAPI(
    title="Machine Learning Calculator",
    description="""
    This is a RESTful API server which applies different machine learning algorithms to make predictions for different tasks.
    """,
    openapi_tags=tags_metadata,
)

app.add_middleware(CORSMiddleware, allow_origins="*", allow_methods="*")
# app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "machine-learning-calculator.herokuapp.com",
        "idrps.shealth.info",
        "localhost",
        "127.0.0.1",
        "0.0.0.0"
    ],
)


@app.get("/", tags=["Root"], description="Redirects to the API docs")
async def root():
    return RedirectResponse("docs")


@app.post(
    "/prostate/bin",
    tags=["Prostate cancer"],
    description="Predict the probability of having prostate cancer",
)
async def predict_prostate_bin(prostate: Prostate, model: Optional[ModelName] = None):
    return predict.prostate_bin(prostate, modelName=model)


@app.post(
    "/prostate/sign",
    tags=["Prostate cancer"],
    description="Predict the probability of having significant prostate cancer",
)
async def predict_prostate_sign(prostate: Prostate, model: Optional[ModelName] = None):
    return predict.prostate_sign(prostate, modelName=model)


@app.get(
    "/models", tags=["Model"], description="Returns a list of machine learning models"
)
async def get_models():
    return {"models": list(ModelName)}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)