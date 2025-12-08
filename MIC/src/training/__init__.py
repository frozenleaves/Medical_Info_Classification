"""训练模块"""
from .trainer import MultiModalTrainer
from .loss import create_loss_function
from .metrics import MultiModalMetrics

__all__ = ['MultiModalTrainer', 'create_loss_function', 'MultiModalMetrics']
