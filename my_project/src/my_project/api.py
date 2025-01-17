import re
from enum import Enum
from http import HTTPStatus

import numpy as np
import onnxruntime
from fastapi import FastAPI


class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


app = FastAPI()


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


@app.get("/predict")
def predict():
    """Predict using ONNX model."""
    # Load the ONNX model
    model = onnxruntime.InferenceSession("models/model.onnx")

    # Prepare the input data
    input_data = {"input": np.random.rand(1, 3).astype(np.float32)}

    # Run the model
    output = model.run(None, input_data)

    return {"output": output[0].tolist()}
