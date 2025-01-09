import math
import random

import numpy as np
import torch


def cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        lr = max(
            min_factor,
            min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)),
        )
        return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_seed(seed: int) -> None:
    """
    Set the seed for random number generation in torch, numpy, and random libraries.

    Args:
        seed (int): The seed value to set for random number generation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
