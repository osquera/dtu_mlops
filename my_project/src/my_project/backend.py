import json
import os
from contextlib import asynccontextmanager

import anyio
import onnxruntime
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application."""
    global model, transform, imagenet_classes
    # Load MNIST model
    model = onnxruntime.InferenceSession(os.path.join(os.getcwd(), "models", "model.onnx"))

    # Load the image transformation pipeline
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    yield


app = FastAPI(lifespan=lifespan)


def predict_image(image_path: str) -> str:
    """Predict image class (or classes) given image path and return the result."""
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0)
    # image shape is (1, 1, 28, 28) and it
    output = model.run(None, {"input": img.numpy()})[0]
    probs = torch.tensor(output).softmax(dim=-1)
    _, predicted_idx = torch.max(probs, 1)
    return probs, predicted_idx.item()


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
