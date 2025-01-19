import os
from contextlib import asynccontextmanager

import anyio
import numpy as np
import onnxruntime
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application."""
    global model, transform, imagenet_classes
    # Load MNIST model
    model = onnxruntime.InferenceSession(os.path.join(os.getcwd(), "models", "model.onnx"))

    yield


app = FastAPI(lifespan=lifespan)


def predict_image(image_path: str) -> str:
    """Predict image class (or classes) given image path and return the result."""
    img = Image.open(image_path).convert("L").resize((28, 28))
    np_img = np.array(img, dtype=np.float32)
    np_img = (np_img / 255.0) * 2.0 - 1.0  # Normalize to [-1, 1]
    np_img = np_img[np.newaxis, np.newaxis, :, :]  # Shape [1,1,28,28]
    output = model.run(None, {"input": np_img})[0]
    exp_out = np.exp(output)
    probabilities = exp_out / np.sum(exp_out, axis=1, keepdims=True)
    prediction = int(np.argmax(probabilities, axis=1)[0])
    return probabilities, prediction


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}


# FastAPI endpoint to classify an image
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        contents = await file.read()
        async with await anyio.open_file(file.filename, "wb") as f:
            await f.write(contents)
        probabilities, prediction = predict_image(file.filename)
        return {
            "filename": file.filename,
            "prediction": prediction,
            "probabilities": probabilities.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
