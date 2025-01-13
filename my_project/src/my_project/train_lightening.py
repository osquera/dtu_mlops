import pytorch_lightning as pl
import torch
from torch import nn
from data import corrupt_mnist


class MyAwesomeModel(pl.LightningModule):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

    def training_step(self, batch):
        """Training step."""
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)
        accuracy = (y_pred.argmax(dim=1) == target).float().mean()
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        self.logger.experiment.log({"train_loss": loss.item(), "train_accuracy": accuracy.item()})
        return loss
    
    def validation_step(self, batch):
        """Validation step."""
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)
        accuracy = (y_pred.argmax(dim=1) == target).float().mean()
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        self.logger.experiment.log({"val_loss": loss.item(), "val_accuracy": accuracy.item()})
        return loss
    
    def test_step(self, batch):
        """Test step."""
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)
        accuracy = (y_pred.argmax(dim=1) == target).float().mean()
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        self.logger.experiment.log({"test_loss": loss.item(), "test_accuracy": accuracy.item()})
        return loss
    

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="mymodel-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    trainer = pl.Trainer(max_epochs=10, limit_train_batches=0.2, callbacks=[checkpoint_callback],
                          logger=pl.loggers.WandbLogger(project="dtu_mlops"),
                            accelerator="gpu", devices=1, precision='bf16-true')

    train_set, test_set = corrupt_mnist()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32)

    trainer.fit(model, train_loader, test_loader)
    trainer.test(model, test_loader)




