"""Adapted from https://github.com/Jackson-Kang/PyTorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""

import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import hydra
from my_project.model_s3 import Decoder, Encoder, Model
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

import logging

log = logging.getLogger(__name__)

# Config parameters
config_path = os.getcwd() + os.sep + "configs" + os.sep + "configs.yaml"

# Model Hyperparameters
dataset_path = os.getcwd() + os.sep + "data"
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 100
x_dim = 784
hidden_dim = 400

# Data loading
mnist_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(
    dataset_path, transform=mnist_transform, train=True, download=True
)
test_dataset = MNIST(
    dataset_path, transform=mnist_transform, train=False, download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

@hydra
def get_model(cfg):
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=cfg.hidden_dim)
    decoder = Decoder(latent_dim=cfg.hidden_dim, hidden_dim=hidden_dim, output_dim=x_dim)

    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)


def loss_function(x, x_hat, mean, log_var):
    """Elbo loss function."""
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + kld


@hydra.main(config_path=config_path)
def train_vae(cfg):
    model = get_model(cfg)
    logging.info("Start training VAE...")
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    model.train()
    for epoch in range(20):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                logging.info(batch_idx)
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        logging.info(
            f"Epoch {epoch+1} complete!,  Average Loss: {overall_loss / (batch_idx*batch_size)}"
        )
    logging.info("Finish!!")

    # save weights
    torch.save(model, f"{os.getcwd()}/trained_model.pt")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx % 100 == 0:
                logging.info(f"Batch {batch_idx}")
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            x_hat, _, _ = model(x)
            break

    save_image(x.view(batch_size, 1, 28, 28), "orig_data.png")
    save_image(x_hat.view(batch_size, 1, 28, 28), "reconstructions.png")

    # Generate samples

    with torch.no_grad():
        noise = torch.randn(batch_size, 20).to(DEVICE)
        generated_images = model.decoder(noise)

    save_image(generated_images.view(batch_size, 1, 28, 28), "generated_sample.png")

if __name__ == "__main__":
    train_vae()
