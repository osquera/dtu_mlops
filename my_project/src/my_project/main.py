import re
from contextlib import asynccontextmanager
from enum import Enum
from http import HTTPStatus
from typing import Optional

import cv2
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor


class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, feature_extractor, tokenizer, device, gen_kwargs
    print("Loading model")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

    yield

    print("Cleaning up")
    del model, feature_extractor, tokenizer, device, gen_kwargs


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
    return {"item_id": item_id}


@app.get("/query_items")
def read_item(item_id: int):
    return {"item_id": item_id}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


database = {"username": [], "password": []}


@app.post("/login/")
def login(username: str, password: str):
    username_db = database["username"]
    password_db = database["password"]
    if username not in username_db and password not in password_db:
        with open("database.csv", "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"


@app.get("/text_model/")
def contains_email(data: str):
    regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data) is not None,
    }
    return response


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...)):
    with open("image.jpg", "wb") as image:
        content = await data.read()
        image.write(content)
        image.close()

    h, w = 28, 28
    img = cv2.imread("image.jpg")
    res = cv2.resize(img, (h, w))
    cv2.imwrite("image_resize.jpg", res)

    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "image": FileResponse("image_resize.jpg", media_type="image/jpg", filename="image.jpg"),
    }
    return response["image"]


@app.post("/caption/")
async def caption(data: UploadFile = File(...)):
    """Generate a caption for an image."""
    i_image = Image.open(data.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [pred.strip() for pred in preds]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="host", port=8000)
