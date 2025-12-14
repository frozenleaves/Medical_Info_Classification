#!/usr/bin/env python3
"""
演示数据扫描功能

使用 sub-dataset/常安OLK+OSF 作为示例
"""

import json
from pathlib import Path
from collections import defaultdict


def scan_demo_data():
    """扫描演示数据"""
    
    # 示例数据路径
    sample_dir = Path("sub-dataset/常安OLK+OSF")
    
    if not sample_dir.exists():
        print("❌ 示例数据不存在")
        return
    
    print("\n" + "=" * 80)
    print("📁 扫描示例数据: sub-dataset/常安OLK+OSF")
    print("=" * 80)
    
    # 扫描照片
    photo_files = []
    photo_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    for ext in photo_extensions:
        photo_files.extend(list(sample_dir.glob(f"*{ext}")))
    
    print(f"\n📷 照片文件 ({len(photo_files)}个):")
    for i, photo in enumerate(sorted(photo_files)[:5], 1):  # 只显示前5个
        print(f"  {i}. {photo.name}")
    if len(photo_files) > 5:
        print(f"  ... 还有 {len(photo_files) - 5} 个")
    
    # 扫描文本
    text_files = list(sample_dir.glob("*.txt"))
    print(f"\n📄 文本文件 ({len(text_files)}个):")
    for i, text in enumerate(text_files, 1):
        print(f"  {i}. {text.name}")
        # 显示前几行
        with open(text, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:3]
        for line in lines:
            print(f"     {line.strip()}")
    
    # 扫描病理切片
    processed_dir = sample_dir / "processed"
    if processed_dir.exists():
        pathology_files = []
        pathology_extensions = [".svs", ".tif", ".tiff", ".kfb"]
        for ext in pathology_extensions:
            pathology_files.extend(list(processed_dir.glob(f"*{ext}")))
        
        print(f"\n🔬 病理切片 ({len(pathology_files)}个):")
        for i, path_file in enumerate(pathology_files, 1):
            size_mb = path_file.stat().st_size / (1024 * 1024)
            print(f"  {i}. {path_file.name} ({size_mb:.1f} MB)")
    
    # 生成样本索引示例
    sample = {
        "id": "常安OLK+OSF/常安",
        "label": "口腔白斑病",  # 示例类别
        "text_path": str(text_files[0].relative_to(sample_dir.parent)) if text_files else None,
        "photo_paths": [
            str(f.relative_to(sample_dir.parent)) for f in sorted(photo_files)
        ],
        "pathology_paths": [
            {
                "path": str(f.relative_to(sample_dir.parent)),
                "roi": None
            } for f in pathology_files
        ]
    }
    
    print("\n" + "=" * 80)
    print("📋 生成的样本索引格式示例:")
    print("=" * 80)
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 80)
    print("✅ 数据扫描完成！")
    print("=" * 80)
    
    print("\n💡 说明:")
    print("  1. 照片文件直接放在样本文件夹根目录")
    print("  2. 文本文件(.txt)放在样本文件夹根目录")
    print("  3. 病理切片放在 processed/ 子目录")
    print("  4. 这个结构符合训练脚本的要求")
    
    print("\n🚀 下一步:")
    print("  1. 组织你的完整数据集:")
    print("     /datasets/")
    print("     ├── 类别1/")
    print("     │   ├── 样本1/  (参照此示例组织)")
    print("     │   └── 样本2/")
    print("     └── 类别2/")
    print("  ")
    print("  2. 运行数据准备脚本:")
    print("     python MIC/src/utils/prepare_dataset_from_folders.py /path/to/datasets")
    print("  ")
    print("  3. 开始训练:")
    print("     python train_from_dataset.py /path/to/datasets")


if __name__ == "__main__":
    scan_demo_data()

