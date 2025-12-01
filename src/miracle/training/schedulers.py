"""
Learning rate schedulers with warmup support.

Implements various warmup strategies for stable training initialization.
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup.

    Gradually increases learning rate from 0 to base_lr over warmup_epochs,
    then applies a base scheduler for the remaining training.
    """

    def __init__(self, optimizer, warmup_epochs, base_scheduler=None, warmup_type='linear'):
        """
        Initialize warmup scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs for warmup
            base_scheduler: Base scheduler to use after warmup (optional)
            warmup_type: Type of warmup ('linear', 'exponential', 'cosine')
        """
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_type = warmup_type
        self.current_epoch = 0

        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        super().__init__(optimizer)

    def get_lr(self):
        """Compute learning rate for current epoch."""
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            if self.warmup_type == 'linear':
                # Linear warmup: lr = base_lr * (epoch / warmup_epochs)
                scale = (self.current_epoch + 1) / self.warmup_epochs
            elif self.warmup_type == 'exponential':
                # Exponential warmup: lr = base_lr * exp(log(0.01) * (1 - epoch/warmup))
                scale = math.exp(math.log(0.01) * (1 - (self.current_epoch + 1) / self.warmup_epochs))
            elif self.warmup_type == 'cosine':
                # Cosine warmup: lr = base_lr * 0.5 * (1 + cos(pi * (1 - epoch/warmup)))
                scale = 0.5 * (1 + math.cos(math.pi * (1 - (self.current_epoch + 1) / self.warmup_epochs)))
            else:
                raise ValueError(f"Unknown warmup_type: {self.warmup_type}")

            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Base scheduler phase
            if self.base_scheduler is not None:
                return self.base_scheduler.get_last_lr()
            else:
                return self.base_lrs

    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            self.current_epoch += 1
        else:
            self.current_epoch = epoch

        # Update optimizer learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        # Step base scheduler if past warmup
        if self.current_epoch >= self.warmup_epochs and self.base_scheduler is not None:
            self.base_scheduler.step()


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine annealing with warmup.

    Combines warmup with cosine annealing for smooth learning rate schedule.
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, warmup_type='linear'):
        """
        Initialize cosine warmup scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of training epochs
            min_lr: Minimum learning rate (for cosine annealing)
            warmup_type: Type of warmup
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_type = warmup_type
        self.current_epoch = 0

        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        super().__init__(optimizer)

    def get_lr(self):
        """Compute learning rate for current epoch."""
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            if self.warmup_type == 'linear':
                scale = (self.current_epoch + 1) / self.warmup_epochs
            elif self.warmup_type == 'exponential':
                scale = math.exp(math.log(0.01) * (1 - (self.current_epoch + 1) / self.warmup_epochs))
            elif self.warmup_type == 'cosine':
                scale = 0.5 * (1 + math.cos(math.pi * (1 - (self.current_epoch + 1) / self.warmup_epochs)))
            else:
                scale = (self.current_epoch + 1) / self.warmup_epochs

            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            self.current_epoch += 1
        else:
            self.current_epoch = epoch

        # Update optimizer learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def get_scheduler(optimizer, scheduler_type, **kwargs):
    """
    Factory function to create scheduler with warmup support.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        **kwargs: Scheduler-specific arguments

    Returns:
        Learning rate scheduler

    Examples:
        # Cosine with warmup
        scheduler = get_scheduler(
            optimizer,
            'cosine_warmup',
            warmup_epochs=10,
            max_epochs=100
        )

        # Linear with warmup then plateau
        scheduler = get_scheduler(
            optimizer,
            'warmup_plateau',
            warmup_epochs=5,
            patience=10
        )
    """
    warmup_epochs = kwargs.pop('warmup_epochs', 0)
    warmup_type = kwargs.pop('warmup_type', 'linear')

    if scheduler_type == 'cosine_warmup':
        max_epochs = kwargs.get('max_epochs', 100)
        min_lr = kwargs.get('min_lr', 0)
        return CosineWarmupScheduler(
            optimizer,
            warmup_epochs,
            max_epochs,
            min_lr,
            warmup_type
        )

    elif scheduler_type == 'warmup_plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        base_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            verbose=kwargs.get('verbose', True)
        )
        return WarmupScheduler(optimizer, warmup_epochs, base_scheduler, warmup_type)

    elif scheduler_type == 'warmup_step':
        from torch.optim.lr_scheduler import StepLR
        base_scheduler = StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
        return WarmupScheduler(optimizer, warmup_epochs, base_scheduler, warmup_type)

    elif scheduler_type == 'warmup_cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 0)
        )
        return WarmupScheduler(optimizer, warmup_epochs, base_scheduler, warmup_type)

    elif scheduler_type == 'warmup_onecycle':
        from torch.optim.lr_scheduler import OneCycleLR
        # OneCycleLR has built-in warmup, so just create it
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 0.01),
            total_steps=kwargs.get('total_steps', 1000),
            pct_start=kwargs.get('pct_start', 0.3),  # 30% warmup
            anneal_strategy=kwargs.get('anneal_strategy', 'cos')
        )

    else:
        # Standard schedulers without warmup
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 100))
        elif scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            return StepLR(optimizer, step_size=kwargs.get('step_size', 30))
        elif scheduler_type == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            return ReduceLROnPlateau(optimizer, patience=kwargs.get('patience', 10))
        elif scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# Example usage
if __name__ == '__main__':
    import torch.nn as nn

    # Create dummy model and optimizer
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Test different warmup schedulers
    print("Testing warmup schedulers...\n")

    # Linear warmup
    scheduler = get_scheduler(
        optimizer,
        'cosine_warmup',
        warmup_epochs=10,
        max_epochs=100,
        warmup_type='linear'
    )

    print("Epoch\tLearning Rate")
    print("-" * 30)
    for epoch in range(20):
        lr = optimizer.param_groups[0]['lr']
        print(f"{epoch}\t{lr:.6f}")
        scheduler.step()
