import json
import logging
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from google.cloud import storage
from pydantic import BaseModel
from starlette.status import HTTP_400_BAD_REQUEST
from xgboost import XGBClassifier

class Island(str, Enum):
    """Valid penguin island options."""
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    """Valid penguin sex options."""
    male = "male"
    female = "female"

class PenguinFeatures(BaseModel):
    """
    Schema for penguin features used in predictions.
    """
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

app = FastAPI()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("penguin-api")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Custom validation error handler to return HTTP 400.
    """
    logger.debug(f"Validation error on {request.url}: {exc}")
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()}
    )

def _download_from_gcs(uri: str) -> Path:
    """
    Download a GCS blob to a temporary file and return its path.

    Args:
        uri (str): The GCS URI in the format 'gs://bucket_name/blob_name'.

    Returns:
        Path: The local path to the downloaded file.
    """
    parts = uri.replace("gs://", "").split("/", 1)
    bucket_name, blob_name = parts[0], parts[1]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    temp_dir = Path(tempfile.mkdtemp())
    destination = temp_dir / Path(blob_name).name
    blob.download_to_filename(str(destination))
    logger.info(f"Downloaded {uri} to {destination}")
    return destination

# ----- Model & Metadata Loading -----
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_MODEL_PATH = PROJECT_ROOT / "app" / "data" / "model.json"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "app" / "data" / "metadata.json"

MODEL_PATH = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
METADATA_PATH = os.environ.get("METADATA_PATH", str(DEFAULT_METADATA_PATH))

model = XGBoostClassifier = XGBClassifier()
if MODEL_PATH.startswith("gs://"):
    local_model = _download_from_gcs(MODEL_PATH)
    model.load_model(str(local_model))
else:
    model.load_model(MODEL_PATH)

if METADATA_PATH.startswith("gs://"):
    local_meta = _download_from_gcs(METADATA_PATH)
    metadata_file = local_meta
else:
    metadata_file = Path(METADATA_PATH)

with open(metadata_file, "r") as f:
    meta: Dict[str, List[str]] = json.load(f)

FEATURE_COLUMNS = meta["feature_columns"]
LABEL_CLASSES = meta["label_classes"]
logger.info(f"Loaded model with {len(FEATURE_COLUMNS)} features and {len(LABEL_CLASSES)} classes")

@app.get("/", include_in_schema=False)
def read_root() -> Dict[str, str]:
    """Root endpoint returning a welcome message."""
    return {"message": "Hello! Welcome to the Penguins Classification API."}

@app.get("/health")
def health_check() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.post("/predict")
def predict(features: PenguinFeatures) -> Dict[str, Any]:
    """
    Predict the species of a penguin from provided features.

    Args:
        features (PenguinFeatures): Validated penguin features.

    Returns:
        dict: Predicted penguin species.
    """
    payload = features.model_dump()
    logger.info(f"Prediction requested: {payload}")

    df = pd.DataFrame([payload])
    df = pd.get_dummies(df, columns=["sex", "island"])
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    try:
        pred = model.predict(df)[0]
        result = LABEL_CLASSES[int(pred)]
        logger.info(f"Prediction result: {result}")
        return {"species": result}
    except Exception as e:
        logger.error("Prediction failed", exc_info=e)
        raise HTTPException(status_code=500, detail="Internal prediction error")
