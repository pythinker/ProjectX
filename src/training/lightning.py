import torch
import lightning.pytorch as pl

from src.training.lr_scheduler import LRScheduler


class LightningModule_(pl.LightningModule):

    def __init__(self, hp, model, loss_fn):

        super().__init__()
        self.automatic_optimization = False
        self.hp = hp
        self.model = model
        self.loss_fn = loss_fn

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), **self.hp.optimizer)
        self.lr_scheduler = LRScheduler(self.hp, optimizer)
        return [optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):

        X, y = batch["X"], batch["y"]

        self.model.train()
        y_pred = self.model(X).squeeze()
        batch_loss, batch_accuracy = self.loss_fn(y_pred, y)

        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(batch_loss)
        optimizer.step()

        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step_()

        self.log_dict({"Training_loss": batch_loss, "Training_accuracy": batch_accuracy, "LR": lr_scheduler.get_lr()}, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return batch_loss

    def validation_step(self, batch, batch_idx):

        X, y = batch["X"], batch["y"]

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).squeeze()
            batch_loss, batch_accuracy = self.loss_fn(y_pred, y)


        self.log_dict({"Validation_loss": batch_loss, "Validation_accuracy": batch_accuracy}, prog_bar=True, logger=True)

        return batch_loss
