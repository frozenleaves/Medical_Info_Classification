"""
MIC Source Package
"""

# 导入主要组件
from .models.multimodal_model import create_multimodal_model
from .training.trainer import MultiModalTrainer
from .data.dataset import MultiModalMedicalDataset
from .data.dataloader import create_dataloaders

__all__ = [
    'create_multimodal_model',
    'MultiModalTrainer',
    'MultiModalMedicalDataset', 
    'create_dataloaders'
]
