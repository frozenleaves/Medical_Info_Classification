"""模型模块"""

from .multimodal_model import create_multimodal_model
from .classifier import create_classifier

__all__ = ["create_multimodal_model", "create_classifier"]
