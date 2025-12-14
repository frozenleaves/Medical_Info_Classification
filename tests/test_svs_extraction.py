"""
SVS 大型病理切片 Patch 提取测试

演示如何使用 OpenSlide 从大型 SVS 文件中提取 patches
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_openslide_availability():
    """测试 OpenSlide 是否可用"""
    print("=" * 80)
    print("检查 OpenSlide 安装状态")
    print("=" * 80)
    
    try:
        import openslide
        print(f"✅ OpenSlide 已安装")
        print(f"   版本: {openslide.__version__}")
        print(f"   库路径: {openslide.__file__}")
        return True
    except ImportError as e:
        print(f"❌ OpenSlide 未安装: {e}")
        print(f"\n安装方法:")
        print(f"  macOS:   brew install openslide && pip install openslide-python")
        print(f"  Ubuntu:  sudo apt-get install openslide-tools python3-openslide")
        print(f"  Windows: 参考 https://openslide.org/download/")
        return False


def test_svs_info(svs_file_path: str):
    """
    显示 SVS 文件信息
    
    Args:
        svs_file_path: SVS 文件路径
    """
    print("\n" + "=" * 80)
    print(f"分析 SVS 文件: {svs_file_path}")
    print("=" * 80)
    
    try:
        import openslide
        
        slide = openslide.OpenSlide(svs_file_path)
        
        print(f"\n📊 基本信息:")
        print(f"  - 层数: {slide.level_count}")
        print(f"  - 厂商: {slide.properties.get('openslide.vendor', 'Unknown')}")
        print(f"  - 扫描倍数: {slide.properties.get('openslide.objective-power', 'Unknown')}x")
        
        print(f"\n📐 各层分辨率（金字塔结构）:")
        for i in range(slide.level_count):
            dims = slide.level_dimensions[i]
            downsample = slide.level_downsamples[i]
            pixels = dims[0] * dims[1]
            
            if i == 0:
                # Level 0 是最高分辨率（原始扫描）
                print(f"  Level {i}: {dims[0]:>6} x {dims[1]:>6} | 最高分辨率 (原始) | {pixels/1e6:>6.1f}M 像素")
            else:
                # 其他 level 是降采样版本
                scale_factor = 1.0 / downsample
                print(f"  Level {i}: {dims[0]:>6} x {dims[1]:>6} | 下采样 {downsample:>4.1f}x (Level 0 的 {scale_factor:.2%}) | {pixels/1e6:>6.1f}M 像素")
        
        print(f"\n🧮 Patch 提取估算 (Level 0):")
        width, height = slide.level_dimensions[0]
        
        for patch_size in [256, 512, 1024]:
            num_x = width // patch_size
            num_y = height // patch_size
            total = num_x * num_y
            print(f"  Patch {patch_size}x{patch_size}: {num_x} x {num_y} = {total:>5} 个patches")
        
        slide.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_patch_extraction():
    """测试 patch 提取功能"""
    print("\n" + "=" * 80)
    print("测试 Patch 提取功能")
    print("=" * 80)
    
    try:
        from MIC.src.data.dataset import MultiModalMedicalDataset
        from MIC.src.data.transforms import PathologyTransforms
        import json
        from datetime import datetime
        
        # 检查是否有真实的 SVS 文件
        data_dir = project_root / 'A_Datasets'
        
        # 创建临时索引（假设有一个 SVS 文件）
        test_svs = data_dir / 'pathology' / 'patient_001' / 'slide1.svs'
        
        if not test_svs.exists():
            # 尝试查找任何 SVS 文件
            svs_files = list((data_dir / 'pathology').rglob('*.svs'))
            if not svs_files:
                svs_files = list((data_dir / 'pathology').rglob('*.tif'))
            
            if not svs_files:
                print("⚠️  未找到 SVS/TIF 文件用于测试")
                print(f"   请将 SVS 文件放在: {data_dir / 'pathology'}")
                return False
            
            test_svs = svs_files[0]
            print(f"✅ 找到测试文件: {test_svs}")
        
        # 先显示文件信息
        test_svs_info(str(test_svs))
        
        # 创建测试索引
        test_index = [
            {
                "id": "test_svs",
                "label": "test",
                "text_path": "texts/patient_001.txt",
                "photo_paths": [],
                "pathology_paths": [str(test_svs.relative_to(data_dir))]
            }
        ]
        
        index_file = data_dir / 'test_svs_index.json'
        with open(index_file, 'w') as f:
            json.dump(test_index, f, indent=2)
        
        # 配置
        config = {
            'patch_size': 512,
            'extract_levels': 'all',
            'overlap': 0.1,
            'max_patches': 10000,  # 提取1000个
            'filter_blank_patches': True,  # 过滤空白区域
        }
        
        # ✅ 重要：PathologyTransforms 也要配置 patch_size=512
        # 否则会默认 resize 到 224×224
        transform_config = {
            'patch_size': 512,  # 与上面的 config 保持一致
            'enable_augment': False  # 测试时不做数据增强
        }
        
        transforms = {
            'pathology': PathologyTransforms(config=transform_config, is_training=False)
        }
        
        print(f"\n🔧 创建 Dataset...")
        print(f"  - Patch 大小: {config['patch_size']}x{config['patch_size']}")
        print(f"  - 最大 Patches: {config['max_patches']}")
        
        dataset = MultiModalMedicalDataset(
            data_dir=str(data_dir),
            split='test_svs',
            transforms=transforms,
            config=config
        )
        
        print(f"\n📦 加载样本...")
        sample = dataset[0]
        
        pathology_data = sample['pathology']
        patches = pathology_data['patches']
        coordinates = pathology_data['coordinates']
        original_size = pathology_data['original_size']
        
        print(f"\n✅ 提取成功!")
        print(f"  - 原始尺寸: {original_size[0]} x {original_size[1]}")
        print(f"  - 提取的 Patches: {len(patches)}")
        if patches:
            print(f"  - Patch 形状: {patches[0].shape}")
            print(f"  - 前几个坐标: {coordinates[:5]}")
        
        # 💾 保存 patches 为 PNG 文件
        if patches:
            save_patches_to_files(
                patches, 
                coordinates, 
                original_size,
                test_svs.stem,  # 使用文件名（不含扩展名）
                config['patch_size']
            )
        
        # 清理
        index_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_patches_to_files(patches, coordinates, original_size, slide_name, patch_size):
    """
    将提取的 patches 保存为 PNG 文件
    
    Args:
        patches: patch tensor 列表
        coordinates: 坐标列表 [(x, y, level), ...] 或 [(x, y), ...]
        original_size: 原始图像尺寸 (width, height)
        slide_name: 切片文件名
        patch_size: patch 尺寸
    """
    print(f"\n" + "=" * 80)
    print("💾 保存 Patches 到文件")
    print("=" * 80)
    
    try:
        from datetime import datetime
        from PIL import Image
        import numpy as np
        
        # 创建输出文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / 'patch_outputs' / f'{slide_name}_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 输出文件夹: {output_dir}")
        print(f"   (相对路径: {output_dir.relative_to(project_root)})")
        
        # 保存每个 patch
        saved_count = 0
        for i, (patch_tensor, coord) in enumerate(zip(patches, coordinates)):
            try:
                # Tensor 格式: [C, H, W]，需要转为 [H, W, C]
                if isinstance(patch_tensor, torch.Tensor):
                    # 反归一化（如果经过了归一化）
                    # 假设使用了 ImageNet 标准归一化
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    
                    # 尝试反归一化
                    patch_denorm = patch_tensor * std + mean
                    # 裁剪到 [0, 1]
                    patch_denorm = torch.clamp(patch_denorm, 0, 1)
                    
                    # 转为 numpy [H, W, C]
                    patch_np = patch_denorm.permute(1, 2, 0).numpy()
                    # 转为 0-255
                    patch_np = (patch_np * 255).astype(np.uint8)
                else:
                    patch_np = np.array(patch_tensor)
                
                # 转为 PIL Image
                patch_img = Image.fromarray(patch_np)
                
                # 文件名：包含索引、坐标和层级信息
                # 坐标可能是 (x, y, level) 或 (x, y)
                if len(coord) == 3:
                    x, y, level = coord
                    filename = f"patch_{i:04d}_x{x:05d}_y{y:05d}_L{level}.png"
                    coord_str = f"({x}, {y}, Level {level})"
                else:
                    x, y = coord
                    filename = f"patch_{i:04d}_x{x:05d}_y{y:05d}.png"
                    coord_str = f"({x}, {y})"
                
                filepath = output_dir / filename
                
                # 保存
                patch_img.save(filepath)
                saved_count += 1
                
                if i == 0:
                    print(f"\n保存示例:")
                    print(f"  文件: {filename}")
                    print(f"  位置: {coord_str}")
                    print(f"  尺寸: {patch_img.size}")
                
            except Exception as e:
                print(f"  ⚠️  保存 patch {i} 失败: {e}")
                continue
        
        # 保存元数据信息
        metadata = {
            "slide_name": slide_name,
            "original_size": {"width": original_size[0], "height": original_size[1]},
            "patch_size": patch_size,
            "total_patches": len(patches),
            "saved_patches": saved_count,
            "timestamp": timestamp,
            "patches": []
        }
        
        # 构建 patches 信息列表（兼容二元组和三元组坐标）
        for i, coord in enumerate(coordinates):
            if len(coord) == 3:
                x, y, level = coord
                patch_info = {
                    "index": i,
                    "filename": f"patch_{i:04d}_x{x:05d}_y{y:05d}_L{level}.png",
                    "position": {"x": x, "y": y, "level": level}
                }
            else:
                x, y = coord
                patch_info = {
                    "index": i,
                    "filename": f"patch_{i:04d}_x{x:05d}_y{y:05d}.png",
                    "position": {"x": x, "y": y}
                }
            metadata["patches"].append(patch_info)
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 保存完成!")
        print(f"  - 保存的 Patches: {saved_count}/{len(patches)}")
        print(f"  - 输出目录: {output_dir}")
        print(f"  - 元数据文件: metadata.json")
        
        # 创建可视化概览（可选）
        create_patch_overview(output_dir, patches[:min(16, len(patches))], coordinates[:min(16, len(patches))])
        
        print(f"\n💡 提示:")
        print(f"  - 使用图片查看器打开: {output_dir}")
        print(f"  - 文件命名格式: patch_序号_x坐标_y坐标_L层级.png (多层级) 或 patch_序号_x坐标_y坐标.png")
        print(f"  - 查看 metadata.json 了解详细信息")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()


def create_patch_overview(output_dir, patches, coordinates):
    """
    创建 patches 的可视化概览图
    
    Args:
        output_dir: 输出目录
        patches: patch tensor 列表（最多16个）
        coordinates: 坐标列表
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt
        
        num_patches = len(patches)
        if num_patches == 0:
            return
        
        # 计算网格布局
        cols = min(4, num_patches)
        rows = (num_patches + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, (patch_tensor, coord) in enumerate(zip(patches, coordinates)):
            row = i // cols
            col = i % cols
            ax = axes[row][col]
            
            # 转换 tensor 为可显示的图像
            if isinstance(patch_tensor, torch.Tensor):
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                patch_denorm = torch.clamp(patch_tensor * std + mean, 0, 1)
                img = patch_denorm.permute(1, 2, 0).numpy()
            else:
                img = np.array(patch_tensor) / 255.0 if np.max(patch_tensor) > 1 else np.array(patch_tensor)
            
            ax.imshow(img)
            # 兼容二元组和三元组坐标
            if len(coord) == 3:
                ax.set_title(f'#{i} L{coord[2]}\n({coord[0]}, {coord[1]})', fontsize=8)
            else:
                ax.set_title(f'#{i}\n({coord[0]}, {coord[1]})', fontsize=8)
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(num_patches, rows * cols):
            row = i // cols
            col = i % cols
            axes[row][col].axis('off')
        
        plt.tight_layout()
        overview_path = output_dir / 'patches_overview.png'
        plt.savefig(overview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  - 概览图: patches_overview.png")
        
    except Exception as e:
        print(f"  ⚠️  创建概览图失败: {e}")


def main():
    """主测试函数"""
    print("\n" + "🔬" * 40)
    print("SVS 大型病理切片 Patch 提取测试")
    print("🔬" * 40 + "\n")
    
    # 测试 OpenSlide
    openslide_ok = test_openslide_availability()
    
    if not openslide_ok:
        print("\n⚠️  OpenSlide 未安装，无法继续测试")
        print("   请先安装 OpenSlide，参考上面的安装方法")
        return
    
    # 测试 Patch 提取
    extraction_ok = test_patch_extraction()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    if extraction_ok:
        print("✅ 所有测试通过!")
        print("\n下一步:")
        print("  1. 将你的 SVS 文件放在 A_Datasets/pathology/ 目录下")
        print("  2. 创建索引文件（train_index.json）")
        print("  3. 配置 patch_size、max_patches 等参数")
        print("  4. 开始训练!")
    else:
        print("❌ 部分测试失败")
        print("   请检查 SVS 文件是否存在，OpenSlide 是否正确安装")


if __name__ == "__main__":
    main()

