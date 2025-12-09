"""
数据加载器模块
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter

from .dataset import MultiModalMedicalDataset
from .transforms import get_transforms


class MultiModalCollator:
    """多模态数据整理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.max_photos = self.config.get('max_photos', 10)
        self.max_patches = self.config.get('max_patches', 1000)
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """整理批次数据"""
        # 分离各模态数据
        texts = [item['text'] for item in batch]
        photos_list = [item['photos'] for item in batch]
        pathology_list = [item['pathology'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        sample_ids = [item['sample_id'] for item in batch]
        
        # 处理文本数据
        text_data = self._collate_texts(texts)
        
        # 处理照片数据
        photo_data = self._collate_photos(photos_list)
        
        # 处理病理数据
        pathology_data = self._collate_pathology(pathology_list)
        
        return {
            'text': text_data,
            'photos': photo_data,
            'pathology': pathology_data,
            'labels': labels,
            'sample_ids': sample_ids
        }
    
    def _collate_texts(self, texts: List[str]) -> Dict[str, Any]:
        """整理文本数据"""
        # 文本长度统计
        text_lengths = [len(text) for text in texts]
        
        return {
            'texts': texts,
            'lengths': torch.tensor(text_lengths, dtype=torch.long)
        }
    
    def _collate_photos(self, photos_list: List[List[torch.Tensor]]) -> Dict[str, Any]:
        """整理照片数据"""
        batch_size = len(photos_list)
        
        # 统计每个样本的图片数量
        photo_counts = [len(photos) for photos in photos_list]
        max_photos_in_batch = min(max(photo_counts) if photo_counts else 0, self.max_photos)
        
        if max_photos_in_batch == 0:
            # 没有图片时创建空tensor
            return {
                'images': torch.zeros(batch_size, 1, 3, 224, 224),
                'counts': torch.zeros(batch_size, dtype=torch.long),
                'masks': torch.zeros(batch_size, 1, dtype=torch.bool)
            }
        
        # 创建padded tensor
        padded_photos = torch.zeros(batch_size, max_photos_in_batch, 3, 224, 224)
        photo_masks = torch.zeros(batch_size, max_photos_in_batch, dtype=torch.bool)
        
        for i, photos in enumerate(photos_list):
            num_photos = min(len(photos), self.max_photos)
            if num_photos > 0:
                # 填充实际图片
                photos_tensor = torch.stack(photos[:num_photos])
                padded_photos[i, :num_photos] = photos_tensor
                photo_masks[i, :num_photos] = True
        
        return {
            'images': padded_photos,
            'counts': torch.tensor(photo_counts, dtype=torch.long),
            'masks': photo_masks
        }
    
    def _collate_pathology(self, pathology_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """整理病理数据"""
        batch_size = len(pathology_list)
        
        # 统计每个样本的patch数量
        patch_counts = [len(item['patches']) for item in pathology_list]
        max_patches_in_batch = min(max(patch_counts) if patch_counts else 0, self.max_patches)
        
        if max_patches_in_batch == 0:
            # 没有patches时创建空tensor
            return {
                'patches': torch.zeros(batch_size, 1, 3, 224, 224),
                'counts': torch.zeros(batch_size, dtype=torch.long),
                'masks': torch.zeros(batch_size, 1, dtype=torch.bool),
                'coordinates': []
            }
        
        # 创建padded tensor
        padded_patches = torch.zeros(batch_size, max_patches_in_batch, 3, 224, 224)
        patch_masks = torch.zeros(batch_size, max_patches_in_batch, dtype=torch.bool)
        coordinates_list = []
        
        for i, pathology_item in enumerate(pathology_list):
            patches = pathology_item['patches']
            coords = pathology_item['coordinates']
            
            num_patches = min(len(patches), self.max_patches)
            if num_patches > 0:
                # 填充实际patches
                patches_tensor = torch.stack(patches[:num_patches])
                padded_patches[i, :num_patches] = patches_tensor
                patch_masks[i, :num_patches] = True
                coordinates_list.append(coords[:num_patches])
            else:
                coordinates_list.append([])
        
        return {
            'patches': padded_patches,
            'counts': torch.tensor(patch_counts, dtype=torch.long),
            'masks': patch_masks,
            'coordinates': coordinates_list
        }


def create_weighted_sampler(dataset: MultiModalMedicalDataset) -> WeightedRandomSampler:
    """创建加权采样器以平衡类别"""
    # 获取所有标签
    labels = [dataset[i]['label'].item() for i in range(len(dataset))]
    
    # 计算类别权重
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # 计算每个类别的权重（逆频率）
    class_weights = {}
    for class_id, count in class_counts.items():
        class_weights[class_id] = total_samples / (len(class_counts) * count)
    
    # 为每个样本分配权重
    sample_weights = [class_weights[label] for label in labels]
    
    print("类别分布:")
    for class_id, count in class_counts.items():
        print(f"  类别 {class_id}: {count} 样本 (权重: {class_weights[class_id]:.3f})")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def create_dataloaders(data_dir: str, 
                      config: Dict,
                      use_weighted_sampling: bool = True) -> Dict[str, DataLoader]:
    """创建数据加载器"""
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        # 确定是否为训练模式
        is_training = (split == 'train')
        
        # 获取变换
        transforms = get_transforms(
            config.get('transforms', {}),
            is_training=is_training
        )
        
        # 创建数据集
        try:
            dataset = MultiModalMedicalDataset(
                data_dir=data_dir,
                split=split,
                transforms=transforms,
                config=config.get('dataset', {})
            )
            
            print(f"{split} 数据集: {len(dataset)} 样本")
            
            # 创建采样器
            sampler = None
            if is_training and use_weighted_sampling and len(dataset) > 0:
                sampler = create_weighted_sampler(dataset)
            
            # 创建collator
            collator = MultiModalCollator(config.get('collator', {}))
            
            # 创建数据加载器
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=config.get('batch_size', 4),
                shuffle=(is_training and sampler is None),
                sampler=sampler,
                num_workers=config.get('num_workers', 4),
                pin_memory=config.get('pin_memory', True),
                collate_fn=collator,
                drop_last=is_training
            )
            
            dataloaders[split] = dataloader
            
        except FileNotFoundError as e:
            print(f"警告: {split} 数据集文件不存在: {e}")
            continue
        except Exception as e:
            print(f"错误: 创建 {split} 数据加载器失败: {e}")
            continue
    
    return dataloaders


def create_inference_dataloader(data_dir: str,
                               samples: List[Dict[str, Any]],
                               config: Dict) -> DataLoader:
    """创建推理数据加载器"""
    
    # 获取推理变换（不包含数据增强）
    transforms = get_transforms(
        config.get('transforms', {}),
        is_training=False
    )
    
    # 创建临时数据集
    dataset = MultiModalMedicalDataset(
        data_dir=data_dir,
        split='inference',  # 特殊标识
        transforms=transforms,
        config=config.get('dataset', {})
    )
    
    # 直接设置样本
    dataset.samples = samples
    
    # 创建collator
    collator = MultiModalCollator(config.get('collator', {}))
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,  # 推理时使用batch_size=1
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collator
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试数据加载器
    from ..config.model_config import ModelConfig
    from ..config.training_config import TrainingConfig
    
    # 测试配置
    config = {
        'batch_size': 2,
        'num_workers': 2,
        'pin_memory': True,
        'transforms': {
            'photo': {'enable_augment': True},
            'pathology': {'enable_augment': True}
        },
        'collator': {
            'max_photos': 5,
            'max_patches': 100
        }
    }
    
    try:
        # 创建数据加载器
        data_dir = "/Users/frozen/PycharmProjects/Medical_Image_Classification/A_Datasets"  # 假设数据目录
        dataloaders = create_dataloaders(data_dir, config)
        
        print(f"创建了 {len(dataloaders)} 个数据加载器")
        
        # 测试一个批次
        for split, dataloader in dataloaders.items():
            if len(dataloader) > 0:
                batch = next(iter(dataloader))
                print(f"\n{split} 批次数据结构:")
                print(f"  文本: {len(batch['text']['texts'])} 条")
                print(f"  照片: {batch['photos']['images'].shape}")
                print(f"  病理: {batch['pathology']['patches'].shape}")
                print(f"  标签: {batch['labels'].shape}")
                break
        
    except Exception as e:
        print(f"测试失败: {e}")
        print("请确保数据文件存在且格式正确")
