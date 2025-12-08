"""
Medical Info Classification (MIC) Package
多模态医学信息分类
"""

__version__ = "1.0.0"
__author__ = "frozenleaves"
__email__ = "example@example.com"

from .src.models.multimodal_model import create_multimodal_model
from .src.training.trainer import MultiModalTrainer
from .src.data.dataset import MultiModalMedicalDataset
from .src.data.dataloader import create_dataloaders

__all__ = [
    'create_multimodal_model',
    'MultiModalTrainer', 
    'MultiModalMedicalDataset',
    'create_dataloaders'
]
