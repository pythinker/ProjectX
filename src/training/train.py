import sys
import os
import argparse
import json

from omegaconf import OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

from src.utils.general_helpers import get_pst_time
from src.dataset.torch_dataset import XorDataset
from src.models.model import Xor
from src.training.lr_scheduler import LRScheduler
from src.training.loss_function import XorLoss
from src.training.lightning import LightningModule_


def train(hp):

    train_dataset = XorDataset(hp, mode='train')
    valid_dataset = XorDataset(hp, mode='valid')
    hp.data.num_samples = len(train_dataset)

    train_dataloader = DataLoader(train_dataset, **hp.dataloader)
    valid_dataloader = DataLoader(valid_dataset, **hp.dataloader)

    model = Xor(hp)

    loss_obj = XorLoss(hp)
    loss_fn = loss_obj.loss_fn

    hp.current_time = get_pst_time()
    tb_log_dir = f"{hp.dir_setting.root_dir}/{hp.logging.tb_log_dir}/{hp.current_time}"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=tb_log_dir)

    lightning_module = LightningModule_(hp, model, loss_fn)

    trainer = pl.Trainer(accelerator=hp.trainer.accelerator, strategy=hp.trainer.strategy, devices=hp.trainer.devices,
                         max_epochs=hp.trainer.max_epochs, log_every_n_steps=hp.logging.log_every_n_steps,
                         check_val_every_n_epoch=hp.logging.check_val_every_n_epoch, logger=tb_logger)

    print(type(hp.logging.ckpt_path))
    if hp.logging.ckpt_path == "None":
        trainer.fit(lightning_module, train_dataloader, valid_dataloader)
    else:
        trainer.fit(lightning_module, train_dataloader, valid_dataloader, ckpt_path=hp.logging.ckpt_path)
