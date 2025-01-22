import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import anyio
import numpy as np
import onnxruntime
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from google.cloud import storage
from image_analysis import calculate_image_characteristics
from PIL import Image
from pydantic import BaseModel


class PredictionOutput(BaseModel):
    """Prediction output class."""

    filename: str
    prediction: int
    probabilities: list[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application."""
    global model, transform, imagenet_classes
    # Load MNIST model
    model = onnxruntime.InferenceSession(os.path.join(os.getcwd(), "models", "model.onnx"))

    yield


app = FastAPI(lifespan=lifespan)


def save_prediction_to_gcp(filepath: str, filename: str, prediction: int, probabilities: list[float]):
    """Save the prediction results to GCP bucket."""
    client = storage.Client()
    bucket = client.bucket("dtu_mlops_osquera")
    time = datetime.now(tz=timezone.utc)

    image_characteristics = calculate_image_characteristics(filepath, rgb=False)

    # Prepare prediction data
    data = {
        "image_characteristics": image_characteristics,
        "filename": filename,
        "prediction": prediction,
        "probabilities": probabilities,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    blob = bucket.blob(f"prediction_{time}.json")
    blob.upload_from_string(json.dumps(data))
    print("Prediction saved to GCP bucket.")


def predict_image(image_path: str):
    """Predict image class (or classes) given image path and return the result."""
    img = Image.open(image_path).convert("L").resize((28, 28))
    np_img = np.array(img, dtype=np.float32)
    np_img = (np_img / 255.0) * 2.0 - 1.0  # Normalize to [-1, 1]
    np_img = np_img[np.newaxis, np.newaxis, :, :]  # Shape [1,1,28,28]
    output = model.run(None, {"input": np_img})[0]
    exp_out = np.exp(output)
    probabilities = exp_out / np.sum(exp_out, axis=1, keepdims=True)
    probabilities = probabilities.squeeze()
    prediction = int(np.argmax(probabilities, axis=0))
    return probabilities, prediction


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}


# FastAPI endpoint to classify an image
@app.post("/classify/", response_model=PredictionOutput)
async def classify_image(background_tasks: BackgroundTasks, image: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        contents = await image.read()
        async with await anyio.open_file(image.filename, "wb") as f:
            await f.write(contents)
        probabilities, prediction = predict_image(image.filename)
        background_tasks.add_task(
            save_prediction_to_gcp,
            filepath=image.filename,
            filename=image.filename,
            prediction=prediction,
            probabilities=probabilities.tolist(),
        )

        return PredictionOutput(filename=image.filename, prediction=prediction, probabilities=probabilities.tolist())
    except Exception as e:
        raise HTTPException(status_code=500) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
