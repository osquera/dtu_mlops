import random
import wandb

wandb.init(project="dtu_mlops")
for _ in range(100):
    wandb.log({"test_metric": random.random()})
