"""
多模态医学数据集类
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path

class MultiModalMedicalDataset(Dataset):
    """
    多模态医学数据集类
    支持病历文本、口腔照片、病理切片的联合加载
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transforms: Optional[Dict] = None,
                 config: Optional[Dict] = None):
        """
        Args:
            data_dir: 数据根目录
            split: 数据集分割 ('train', 'val', 'test')
            transforms: 数据变换字典
            config: 配置参数
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transforms = transforms or {}
        self.config = config or {}
        
        # 加载数据索引
        self.samples = self._load_data_index()
        
        # 类别标签映射
        self.label_map = self._create_label_mapping()
        self.num_classes = len(self.label_map)
        
    def _load_data_index(self) -> List[Dict]:
        """加载数据索引文件"""
        index_file = self.data_dir / f"{self.split}_index.json"
        
        if not index_file.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_file}")
            
        with open(index_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
            
        print(f"加载 {self.split} 数据集: {len(samples)} 个样本")
        return samples
    
    def _create_label_mapping(self) -> Dict[str, int]:
        """创建标签映射"""
        # 从所有样本中收集标签
        all_labels = set()
        for sample in self.samples:
            all_labels.add(sample['label'])
        
        # 创建标签到索引的映射
        label_map = {label: idx for idx, label in enumerate(sorted(all_labels))}
        print(f"类别映射: {label_map}")
        return label_map
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        sample_info = self.samples[idx]
        
        # 加载各模态数据
        text_data = self._load_text(sample_info)
        photo_data = self._load_photos(sample_info)
        pathology_data = self._load_pathology(sample_info)
        
        # 获取标签
        label = self.label_map[sample_info['label']]
        
        return {
            'text': text_data,
            'photos': photo_data,
            'pathology': pathology_data,
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': sample_info.get('id', idx)
        }
    
    def _load_text(self, sample_info: Dict) -> str:
        """加载病历文本数据"""
        text_file = self.data_dir / sample_info['text_path']
        
        if not text_file.exists():
            print(f"警告: 文本文件不存在 {text_file}")
            return ""
            
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # 应用文本变换
        if 'text' in self.transforms:
            text = self.transforms['text'](text)
            
        return text
    
    def _load_photos(self, sample_info: Dict) -> List[torch.Tensor]:
        """加载口腔照片数据"""
        photo_paths = sample_info.get('photo_paths', [])
        photos = []
        
        for photo_path in photo_paths:
            img_file = self.data_dir / photo_path
            
            if not img_file.exists():
                print(f"警告: 图片文件不存在 {img_file}")
                continue
                
            try:
                # 加载图片
                image = Image.open(img_file).convert('RGB')
                
                # 应用图片变换
                if 'photo' in self.transforms:
                    image = self.transforms['photo'](image)
                    
                photos.append(image)
                
            except Exception as e:
                print(f"警告: 加载图片失败 {img_file}: {e}")
                continue
        
        # 如果没有有效图片，创建空白图片
        if not photos and 'photo' in self.transforms:
            blank_image = Image.new('RGB', (224, 224), (128, 128, 128))
            photos.append(self.transforms['photo'](blank_image))
        
        return photos
    
    def _load_pathology(self, sample_info: Dict) -> Dict[str, Any]:
        """加载病理切片数据"""
        pathology_paths = sample_info.get('pathology_paths', [])
        pathology_data = {
            'patches': [],
            'coordinates': [],
            'original_size': None
        }
        
        for path_info in pathology_paths:
            if isinstance(path_info, str):
                # 简单路径格式
                wsi_file = self.data_dir / path_info
                patches, coords, original_size = self._extract_patches_from_wsi(wsi_file)
            else:
                # 详细信息格式
                wsi_file = self.data_dir / path_info['path']
                patches, coords, original_size = self._extract_patches_from_wsi(
                    wsi_file, path_info.get('roi', None)
                )
            
            pathology_data['patches'].extend(patches)
            pathology_data['coordinates'].extend(coords)
            if pathology_data['original_size'] is None:
                pathology_data['original_size'] = original_size
        
        return pathology_data
    
    def _extract_patches_from_wsi(self, 
                                  wsi_file: Path, 
                                  roi: Optional[Dict] = None) -> Tuple[List, List, Tuple]:
        """从WSI中提取patches"""
        if not wsi_file.exists():
            print(f"警告: WSI文件不存在 {wsi_file}")
            return [], [], (0, 0)
        
        try:
            # 加载WSI图像
            wsi_image = Image.open(wsi_file).convert('RGB')
            original_size = wsi_image.size
            
            # 获取patch配置
            patch_size = self.config.get('patch_size', 256)
            overlap = self.config.get('overlap', 0.1)
            max_patches = self.config.get('max_patches', 1000)
            
            # 计算步长
            stride = int(patch_size * (1 - overlap))
            
            patches = []
            coordinates = []
            
            # 确定提取区域
            if roi:
                x_start, y_start = roi.get('x', 0), roi.get('y', 0)
                width, height = roi.get('width', original_size[0]), roi.get('height', original_size[1])
                x_end, y_end = x_start + width, y_start + height
            else:
                x_start, y_start = 0, 0
                x_end, y_end = original_size
            
            # 提取patches
            for y in range(y_start, y_end - patch_size + 1, stride):
                for x in range(x_start, x_end - patch_size + 1, stride):
                    if len(patches) >= max_patches:
                        break
                    
                    # 提取patch
                    patch = wsi_image.crop((x, y, x + patch_size, y + patch_size))
                    
                    # 过滤空白patch（可选）
                    if self._is_valid_patch(patch):
                        # 应用变换
                        if 'pathology' in self.transforms:
                            patch = self.transforms['pathology'](patch)
                        
                        patches.append(patch)
                        coordinates.append((x, y))
                
                if len(patches) >= max_patches:
                    break
            
            print(f"从 {wsi_file.name} 提取了 {len(patches)} 个patches")
            return patches, coordinates, original_size
            
        except Exception as e:
            print(f"错误: 处理WSI文件失败 {wsi_file}: {e}")
            return [], [], (0, 0)
    
    def _is_valid_patch(self, patch: Image.Image, threshold: float = 0.1) -> bool:
        """检查patch是否有效（非空白区域）"""
        # 转换为numpy数组
        patch_array = np.array(patch)
        
        # 计算标准差，过滤掉过于单调的patch
        std = np.std(patch_array)
        return std > threshold * 255
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        class_counts = {}
        for sample in self.samples:
            label = sample['label']
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts


def create_data_index_template():
    """创建数据索引文件模板"""
    template = [
        {
            "id": "patient_001",
            "label": "class_A",
            "text_path": "texts/patient_001.txt",
            "photo_paths": [
                "photos/patient_001_photo1.png",
                "photos/patient_001_photo2.png"
            ],
            "pathology_paths": [
                {
                    "path": "pathology/patient_001_slide1.tiff",
                    "roi": {"x": 0, "y": 0, "width": 4000, "height": 4000}
                }
            ]
        }
        # 更多样本...
    ]
    
    return template


if __name__ == "__main__":
    # 创建示例数据索引
    template = create_data_index_template()
    print("数据索引模板:")
    print(json.dumps(template, indent=2, ensure_ascii=False))
