from torch.optim.lr_scheduler import LambdaLR

import math


def get_cosine_schedule_with_warmup(
    optimizer,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    num_warmup_steps=0,
    last_epoch=-1,
):
    """Get learning rate schedule with linear warmup and cosine decay.

    Args:
        optimizer (torch.optim.Optimizer): optimizer.
        num_training_steps (int): total number of training steps.
        num_cycles (float): number of cycles in the cosine decay.
        num_warmup_steps (int): number of warmup steps.
        last_epoch (int): last epoch number.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: learning rate scheduler.
    """

    def _lr_lambda(current_step):
        """
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        """

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)
