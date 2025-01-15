import os
import time

import dotenv
import torch

import wandb
from my_project.model import MyAwesomeModel

dotenv.load_dotenv(".env/.env")

logdir = "wandb_artifacts"


def load_model(model_checkpoint: str) -> MyAwesomeModel:
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={
            "entity": os.getenv("WANDB_ENTITY"),
            "project": os.getenv("WANDB_PROJECT"),
        },
    )
    artifact = api.artifact(model_checkpoint, type="model")
    artifact.download(root=logdir)
    file_name = artifact.files()[0].name
    model = torch.load(f"{logdir}/{file_name}")
    structured_model = MyAwesomeModel()
    structured_model.load_state_dict(model)
    return structured_model


def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 1, 28, 28))
    end = time.time()
    assert end - start < 1


if __name__ == "__main__":
    test_model_speed()
