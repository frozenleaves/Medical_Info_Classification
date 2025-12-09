"""
数据变换模块
"""

import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import re
from typing import Any, List, Dict
import numpy as np
from PIL import Image

class TextTransforms:
    """文本数据变换"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.max_length = self.config.get('max_length', 512)
        self.enable_augment = self.config.get('enable_augment', False)
    
    def __call__(self, text: str) -> str:
        """应用文本变换"""
        # 基础清洗
        text = self._basic_clean(text)
        
        # 数据增强（仅训练时）
        if self.enable_augment and random.random() < 0.3:
            text = self._augment_text(text)
        
        # 截断处理
        text = self._truncate_text(text)
        
        return text
    
    def _basic_clean(self, text: str) -> str:
        """基础文本清洗"""
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留中文、英文、数字、常用标点）
        text = re.sub(r'[^\u4e00-\u9fff\w\s.,;:!?()（），。；：！？]', '', text)
        
        return text.strip()
    
    def _truncate_text(self, text: str) -> str:
        """截断文本到最大长度"""
        if len(text) > self.max_length:
            # 尝试在句号处截断
            sentences = text.split('。')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '。') <= self.max_length:
                    truncated += sentence + '。'
                else:
                    break
            
            if not truncated:  # 如果没有句号，直接截断
                truncated = text[:self.max_length]
            
            return truncated
        
        return text
    
    def _augment_text(self, text: str) -> str:
        """文本数据增强"""
        techniques = self.config.get('augment_techniques', ['synonym_replace'])
        
        for technique in techniques:
            if technique == 'synonym_replace':
                text = self._synonym_replace(text)
            elif technique == 'sentence_shuffle':
                text = self._sentence_shuffle(text)
        
        return text
    
    def _synonym_replace(self, text: str, prob: float = 0.1) -> str:
        """同义词替换（简化版）"""
        # 医学术语同义词映射
        synonyms = {
            '疼痛': '痛楚',
            '红肿': '发红肿胀',
            '溃疡': '溃烂',
            '发炎': '炎症',
            '出血': '流血'
        }
        
        for original, replacement in synonyms.items():
            if original in text and random.random() < prob:
                text = text.replace(original, replacement)
        
        return text
    
    def _sentence_shuffle(self, text: str) -> str:
        """句子顺序打乱"""
        sentences = text.split('。')
        if len(sentences) > 2:
            # 随机交换两个句子
            idx1, idx2 = random.sample(range(len(sentences)), 2)
            sentences[idx1], sentences[idx2] = sentences[idx2], sentences[idx1]
        
        return '。'.join(sentences)


class PhotoTransforms:
    """口腔照片变换"""
    
    def __init__(self, config: Dict = None, is_training: bool = True):
        self.config = config or {}
        self.is_training = is_training
        
        # ✅ 支持可配置的输入尺寸（默认224，支持更高分辨率）
        self.input_size = self.config.get('input_size', 224)
        # 对于高分辨率图像，可以设置为 384, 448, 或 512
        # 注意：更大的尺寸需要更多显存和计算时间
        
        # 基础变换
        base_transforms = [
            A.Resize(self.input_size, self.input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        # 训练时的增强变换
        if is_training and self.config.get('enable_augment', True):
            # 训练时先resize到稍大尺寸，再random crop到目标尺寸
            aug_size = int(self.input_size * 1.143)  # 例如 224 → 256, 384 → 438
            aug_transforms = [
                A.Resize(aug_size, aug_size),
                A.RandomCrop(self.input_size, self.input_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]
            self.transform = A.Compose(aug_transforms)
        else:
            self.transform = A.Compose(base_transforms)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """应用图像变换"""
        # 转换为numpy数组
        image_array = np.array(image)
        
        # 应用albumentations变换
        transformed = self.transform(image=image_array)
        
        return transformed['image']


class PathologyTransforms:
    """病理切片变换（用于单个patch）"""
    
    def __init__(self, config: Dict = None, is_training: bool = True):
        self.config = config or {}
        self.is_training = is_training
        
        # ✅ 支持可配置的patch尺寸（默认224）
        self.patch_size = self.config.get('patch_size', 224)
        # 注意：这里的尺寸是指每个 patch resize 后的尺寸
        # 原始 WSI 的 patch 提取尺寸在 dataset.py 中配置
        
        # 病理图像的特殊预处理
        base_transforms = [
            A.Resize(self.patch_size, self.patch_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        # 训练时的增强（针对病理图像优化）
        if is_training and self.config.get('enable_augment', True):
            aug_size = int(self.patch_size * 1.143)
            aug_transforms = [
                A.Resize(aug_size, aug_size),
                A.RandomCrop(self.patch_size, self.patch_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                # 病理图像特有的增强
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]
            self.transform = A.Compose(aug_transforms)
        else:
            self.transform = A.Compose(base_transforms)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """应用病理图像变换"""
        # 转换为numpy数组
        image_array = np.array(image)
        
        # 应用变换
        transformed = self.transform(image=image_array)
        
        return transformed['image']


class MultiModalTransforms:
    """多模态联合变换"""
    
    def __init__(self, config: Dict = None, is_training: bool = True):
        self.config = config or {}
        self.is_training = is_training
        
        # 创建各模态变换
        self.text_transform = TextTransforms(config.get('text', {}))
        self.photo_transform = PhotoTransforms(config.get('photo', {}), is_training)
        self.pathology_transform = PathologyTransforms(config.get('pathology', {}), is_training)
        
        # 模态dropout概率
        self.modal_dropout_prob = config.get('modal_dropout', 0.1) if is_training else 0.0
    
    def get_transforms(self) -> Dict[str, Any]:
        """获取变换字典"""
        transforms = {
            'text': self.text_transform,
            'photo': self.photo_transform,
            'pathology': self.pathology_transform
        }
        
        return transforms
    
    def apply_modal_dropout(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """应用模态dropout增强"""
        if not self.is_training or random.random() > self.modal_dropout_prob:
            return data
        
        # 随机选择要dropout的模态（至少保留一个）
        available_modals = ['text', 'photos', 'pathology']
        modal_to_drop = random.choice(available_modals)
        
        # 应用dropout
        if modal_to_drop == 'text':
            data['text'] = ""
        elif modal_to_drop == 'photos':
            data['photos'] = []
        elif modal_to_drop == 'pathology':
            data['pathology']['patches'] = []
        
        return data


def get_transforms(config: Dict, is_training: bool = True) -> Dict[str, Any]:
    """获取变换配置"""
    multimodal_transforms = MultiModalTransforms(config, is_training)
    return multimodal_transforms.get_transforms()


if __name__ == "__main__":
    # 测试变换
    from PIL import Image
    
    # 测试图像变换
    config = {
        'photo': {'enable_augment': True},
        'pathology': {'enable_augment': True}
    }
    
    transforms = get_transforms(config, is_training=True)
    
    # 创建测试图像
    test_image = Image.new('RGB', (512, 512), (128, 128, 128))
    
    # 测试照片变换
    photo_tensor = transforms['photo'](test_image)
    print(f"照片变换结果: {photo_tensor.shape}")
    
    # 测试病理变换
    pathology_tensor = transforms['pathology'](test_image)
    print(f"病理变换结果: {pathology_tensor.shape}")
    
    # 测试文本变换
    test_text = "患者口腔内出现红肿。疼痛明显，建议进一步检查。"
    transformed_text = transforms['text'](test_text)
    print(f"文本变换结果: {transformed_text}")
