import abc
from abc import ABC
from typing import Any, Dict, Optional

from src.utils.lr_calculator import linear_warmup_with_cosine_annealing


class LRScheduler:

    def __init__(self, hp, optimizer):

        self.optimizer = optimizer
        self.global_step = 0
        self.min_lr = hp.lr_scheduler.min_lr
        self.max_lr = hp.optimizer.lr
        self.num_mini_batches_per_epoch = round(hp.data.num_samples/hp.dataloader.batch_size)
        self.max_steps = round(hp.trainer.max_epochs * self.num_mini_batches_per_epoch)
        self.warmup_steps = round(hp.lr_scheduler.warmup_ratio * self.max_steps)
        self.decay_steps = round(hp.lr_scheduler.decay_ratio * self.max_steps)

    def step_(self):

        new_lr = linear_warmup_with_cosine_annealing(self.max_lr, self.warmup_steps, self.global_step, self.decay_steps, self.min_lr)
        self.set_lr(new_lr)
        self.global_step += 1

    def get_lr(self):

        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    
    def set_lr(self, new_lr):

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
