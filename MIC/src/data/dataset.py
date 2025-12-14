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

# 尝试导入 OpenSlide（用于处理大型 SVS 文件）
try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("警告: OpenSlide 未安装，将使用 PIL 处理病理图像（不推荐用于大型 SVS 文件）")
    print("安装命令: pip install openslide-python")

class MultiModalMedicalDataset(Dataset):
    """
    多模态医学数据集类
    支持病历文本、照片、病理切片的联合加载
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
                # 提取时不指定level，会根据配置自动处理多level
                patches, coords, original_size = self._extract_patches_from_wsi(wsi_file)
            else:
                # 详细信息格式
                wsi_file = self.data_dir / path_info['path']
                # 可以在索引中指定要提取的 levels
                target_levels = path_info.get('levels', None)
                patches, coords, original_size = self._extract_patches_from_wsi(
                    wsi_file, 
                    roi=path_info.get('roi', None),
                    target_levels=target_levels
                )
            
            pathology_data['patches'].extend(patches)
            pathology_data['coordinates'].extend(coords)
            if pathology_data['original_size'] is None:
                pathology_data['original_size'] = original_size
        
        return pathology_data
    
    def _extract_patches_from_wsi(self, 
                                  wsi_file: Path, 
                                  roi: Optional[Dict] = None,
                                  target_levels: Optional[Any] = None) -> Tuple[List, List, Tuple]:
        """
        从WSI（Whole Slide Image）中提取patches
        
        支持：
        - 大型 SVS 文件（使用 OpenSlide，推荐）
        - 普通图像文件（使用 PIL，作为回退方案）
        - 多层级提取（将每个 level 当作独立切片）
        
        Args:
            wsi_file: WSI文件路径
            roi: 感兴趣区域（可选），格式 {'x': x, 'y': y, 'width': w, 'height': h}
            target_levels: 要提取的层级
                - None: 使用配置中的设置
                - 'all': 提取所有层级
                - [0, 1, 2]: 提取指定层级列表
                - 0: 只提取 level 0（单个整数）
            
        Returns:
            patches: patch 列表
            coordinates: 坐标列表 [(x, y, level), ...]（包含layer信息）
            original_size: 原始图像尺寸 (width, height)（level 0的尺寸）
        """
        if not wsi_file.exists():
            print(f"警告: WSI文件不存在 {wsi_file}")
            return [], [], (0, 0)
        
        # 检查文件扩展名，判断是否是 SVS/大型病理切片文件
        file_ext = wsi_file.suffix.lower()
        is_svs_file = file_ext in ['.svs', '.tif', '.tiff', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.bif']
        
        # 优先使用 OpenSlide 处理 SVS 文件
        if is_svs_file and OPENSLIDE_AVAILABLE:
            return self._extract_patches_with_openslide(wsi_file, roi, target_levels)
        else:
            # 回退到 PIL 处理（适用于小文件）
            if is_svs_file and not OPENSLIDE_AVAILABLE:
                print(f"警告: {wsi_file.name} 是大型病理切片文件，但 OpenSlide 未安装")
                print("      将使用 PIL 处理，可能会消耗大量内存或失败")
            return self._extract_patches_with_pil(wsi_file, roi)
    
    def _extract_patches_with_openslide(self, 
                                       wsi_file: Path, 
                                       roi: Optional[Dict] = None,
                                       target_levels: Optional[Any] = None) -> Tuple[List, List, Tuple]:
        """
        使用 OpenSlide 从大型 SVS 文件中高效提取 patches
        
        优势：
        - 内存高效：按需读取，不会一次性加载整个图像
        - 支持多层金字塔结构
        - 处理超大文件（如 20000x10000 像素）
        - 支持多尺度提取（每个level当作独立切片）
        """
        try:
            # 打开 SVS 文件
            slide = openslide.OpenSlide(str(wsi_file))
            
            # 获取基本信息
            original_size = slide.level_dimensions[0]  # Level 0 的尺寸
            
            print(f"打开 {wsi_file.name}:")
            print(f"  - 层数: {slide.level_count}")
            print(f"  - Level 0 (最高分辨率): {original_size[0]} x {original_size[1]}")
            if slide.level_count > 1:
                for i in range(1, min(slide.level_count, 3)):
                    print(f"  - Level {i}: {slide.level_dimensions[i][0]} x {slide.level_dimensions[i][1]}")
            
            # 决定要提取哪些 levels
            if target_levels is None:
                # 从配置中读取
                extract_levels_config = self.config.get('extract_levels', 0)
                if extract_levels_config == 'all':
                    levels_to_extract = list(range(slide.level_count))
                elif isinstance(extract_levels_config, list):
                    levels_to_extract = extract_levels_config
                else:
                    levels_to_extract = [extract_levels_config]
            elif target_levels == 'all':
                levels_to_extract = list(range(slide.level_count))
            elif isinstance(target_levels, list):
                levels_to_extract = target_levels
            else:
                levels_to_extract = [target_levels]
            
            # 过滤掉不存在的 level
            levels_to_extract = [l for l in levels_to_extract if 0 <= l < slide.level_count]
            
            if not levels_to_extract:
                levels_to_extract = [0]  # 至少提取 level 0
            
            print(f"  - 提取层级: {levels_to_extract}")
            
            # 获取 patch 配置
            patch_size = self.config.get('patch_size', 512)
            overlap = self.config.get('overlap', 0.1)
            max_patches_per_level = self.config.get('max_patches_per_level', None)
            max_patches_total = self.config.get('max_patches', 1000)
            filter_blank = self.config.get('filter_blank_patches', False)
            
            # 如果没指定每层最大数量，则平均分配
            if max_patches_per_level is None:
                max_patches_per_level = max_patches_total // len(levels_to_extract)
            
            # 计算步长
            stride = int(patch_size * (1 - overlap))
            
            # 收集所有 levels 的 patches
            all_patches = []
            all_coordinates = []
            
            # 对每个 level 分别提取
            for level in levels_to_extract:
                level_size = slide.level_dimensions[level]
                
                # 确定提取区域（考虑 level 的缩放）
                if roi:
                    # ROI 坐标是针对 level 0 的，需要缩放到当前 level
                    downsample = slide.level_downsamples[level]
                    x_start = int(roi.get('x', 0) / downsample)
                    y_start = int(roi.get('y', 0) / downsample)
                    x_end = int((roi.get('x', 0) + roi.get('width', original_size[0])) / downsample)
                    y_end = int((roi.get('y', 0) + roi.get('height', original_size[1])) / downsample)
                else:
                    x_start, y_start = 0, 0
                    x_end, y_end = level_size
                
                # 确保不超出边界
                x_end = min(x_end, level_size[0])
                y_end = min(y_end, level_size[1])
                
                print(f"  - Level {level}: 提取区域 ({x_start},{y_start}) 到 ({x_end},{y_end}), Patch={patch_size}x{patch_size}, 步长={stride}")
                
                level_patches = []
                level_coords = []
                total_patches = 0
                filtered_patches = 0
                
                # 生成提取位置列表（包含边缘补充）
                x_positions = list(range(x_start, x_end - patch_size + 1, stride))
                y_positions = list(range(y_start, y_end - patch_size + 1, stride))
                
                # 补充右边缘：如果最后一个位置+patch_size未到边界，添加边界位置
                if x_positions and (x_positions[-1] + patch_size < x_end):
                    # 确保至少有 patch_size 的空间
                    if x_end >= patch_size:
                        x_positions.append(x_end - patch_size)
                
                # 补充下边缘
                if y_positions and (y_positions[-1] + patch_size < y_end):
                    if y_end >= patch_size:
                        y_positions.append(y_end - patch_size)
                
                # 如果区域太小，至少尝试提取一个 patch
                if not x_positions and x_end >= patch_size:
                    x_positions = [x_start]
                if not y_positions and y_end >= patch_size:
                    y_positions = [y_start]
                
                print(f"    X位置数: {len(x_positions)}, Y位置数: {len(y_positions)}, 预计: {len(x_positions) * len(y_positions)} 个patches")
                
                # 在当前 level 上提取 patches
                for y in y_positions:
                    for x in x_positions:
                        if len(level_patches) >= max_patches_per_level:
                            break
                        
                        total_patches += 1
                        
                        # 读取 patch（注意：location 坐标需要转换到 level 0 坐标系）
                        downsample = slide.level_downsamples[level]
                        location_level0 = (int(x * downsample), int(y * downsample))
                        
                        patch_rgba = slide.read_region(
                            location=location_level0,  # level 0 坐标系
                            level=level,               # 但从这个 level 读取
                            size=(patch_size, patch_size)
                        )
                        
                        # 转换为 RGB
                        patch_rgb = patch_rgba.convert('RGB')
                        
                        # 过滤空白 patch
                        if filter_blank and not self._is_valid_patch(patch_rgb):
                            filtered_patches += 1
                            continue
                        
                        # 应用变换
                        if 'pathology' in self.transforms:
                            patch_transformed = self.transforms['pathology'](patch_rgb)
                        else:
                            patch_array = np.array(patch_rgb).astype(np.float32) / 255.0
                            patch_transformed = torch.from_numpy(patch_array).permute(2, 0, 1)
                        
                        level_patches.append(patch_transformed)
                        # 坐标格式：(x, y, level)
                        level_coords.append((x, y, level))
                    
                    if len(level_patches) >= max_patches_per_level:
                        break
                
                print(f"    ✅ Level {level} 提取: {len(level_patches)} 个patches (总计:{total_patches}, 过滤:{filtered_patches})")
                
                all_patches.extend(level_patches)
                all_coordinates.extend(level_coords)
                
                # 检查总数限制
                if len(all_patches) >= max_patches_total:
                    print(f"  ⚠️  达到总数限制 ({max_patches_total})，停止提取")
                    all_patches = all_patches[:max_patches_total]
                    all_coordinates = all_coordinates[:max_patches_total]
                    break
            
            # 关闭 slide
            slide.close()
            
            print(f"  ✅ 总共提取: {len(all_patches)} 个patches（来自 {len(levels_to_extract)} 个层级）")
            
            return all_patches, all_coordinates, original_size
            
        except Exception as e:
            print(f"错误: OpenSlide 处理失败 {wsi_file}: {e}")
            print(f"      回退到 PIL 处理...")
            return self._extract_patches_with_pil(wsi_file, roi)
    
    def _extract_patches_with_pil(self, 
                                  wsi_file: Path, 
                                  roi: Optional[Dict] = None) -> Tuple[List, List, Tuple]:
        """
        使用 PIL 提取 patches（回退方案，适用于小文件）
        
        警告：不适合大型 SVS 文件，会消耗大量内存
        """
        try:
            # 加载整个图像到内存（不适合大文件！）
            wsi_image = Image.open(wsi_file).convert('RGB')
            original_size = wsi_image.size
            
            print(f"使用 PIL 处理 {wsi_file.name} (尺寸: {original_size[0]} x {original_size[1]})")
            if original_size[0] * original_size[1] > 100_000_000:  # 100M 像素
                print(f"  ⚠️  警告: 图像很大，建议安装 OpenSlide 以提高效率")
            
            # 获取 patch 配置
            patch_size = self.config.get('patch_size', 512)
            overlap = self.config.get('overlap', 0.0)
            max_patches = self.config.get('max_patches', 1000)
            filter_blank = self.config.get('filter_blank_patches', True)
            
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
            
            # 提取 patches
            for y in range(y_start, y_end - patch_size + 1, stride):
                for x in range(x_start, x_end - patch_size + 1, stride):
                    if len(patches) >= max_patches:
                        break
                    
                    # 提取 patch
                    patch = wsi_image.crop((x, y, x + patch_size, y + patch_size))
                    
                    # 过滤空白 patch
                    if filter_blank and not self._is_valid_patch(patch):
                        continue
                    
                        # 应用变换
                        if 'pathology' in self.transforms:
                            patch = self.transforms['pathology'](patch)
                        
                        patches.append(patch)
                        coordinates.append((x, y))
                
                if len(patches) >= max_patches:
                    break
            
            print(f"  ✅ 提取了 {len(patches)} 个patches")
            return patches, coordinates, original_size
            
        except Exception as e:
            print(f"错误: PIL 处理失败 {wsi_file}: {e}")
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
