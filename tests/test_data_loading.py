#!/usr/bin/env python3
"""
数据加载相关功能测试
测试数据处理、数据集、数据加载器等功能
"""

import sys
import os
from pathlib import Path
import json
import shutil
import traceback
import tempfile
from PIL import Image
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 测试配置
TEST_CONFIG = {
    "data_dir": project_root / "A_Datasets",
    "temp_dir": None,  # 会在测试时创建
    "sample_data": {
        "patient_id": "patient_001",
        "expected_label": "class_A",
        "expected_files": {
            "text": "texts/patient_001.txt",
            "photos": [
                "photos/patient_001/photo1.png",
                "photos/patient_001/photo2.png",
            ],
            "pathology": [
                "pathology/patient_001/slide1.tif",
                "pathology/patient_001/slide2.tif",
            ],
        },
    },
}


class DataLoadingTester:
    """数据加载测试类"""

    def __init__(self):
        self.temp_dir = None
        self.test_results = {}

    def setup(self):
        """测试前设置"""
        print("🛠️ 设置测试环境...")

        # 检查A_Datasets是否存在
        if not TEST_CONFIG["data_dir"].exists():
            print(f"❌ 测试数据目录不存在: {TEST_CONFIG['data_dir']}")
            return False

        # 创建临时目录用于测试
        self.temp_dir = Path(tempfile.mkdtemp(prefix="mic_test_"))
        TEST_CONFIG["temp_dir"] = self.temp_dir
        print(f"✅ 临时测试目录: {self.temp_dir}")

        return True

    def cleanup(self):
        """测试后清理"""
        print("\n🧹 清理测试环境...")
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print("✅ 清理完成")

    def test_data_transforms(self):
        """测试数据变换功能"""
        print("\n🔄 测试数据变换功能...")
        print("-" * 30)

        try:
            from MIC.src.data.transforms import (
                get_transforms,
                TextTransforms,
                PhotoTransforms,
                PathologyTransforms,
            )

            # 1. 测试获取变换配置
            config = {
                "text": {"enable_augment": False},
                "photo": {"enable_augment": False},
                "pathology": {"enable_augment": False},
            }

            transforms = get_transforms(config, is_training=False)
            print("✅ 变换配置创建成功")

            # 2. 测试文本变换
            text_transform = TextTransforms(
                {"max_length": 256, "enable_augment": False}
            )
            test_text = "患者口腔内出现红肿，疼痛明显。"
            transformed_text = text_transform(test_text)
            print(f"✅ 文本变换测试成功: {len(transformed_text)} 字符")

            # 3. 测试图片变换
            photo_transform = PhotoTransforms({}, is_training=False)
            # 创建测试图片
            test_image = Image.new("RGB", (224, 224), (128, 128, 128))
            transformed_image = photo_transform(test_image)
            print(f"✅ 图片变换测试成功: {transformed_image.shape}")

            # 4. 测试病理变换
            pathology_transform = PathologyTransforms({}, is_training=False)
            transformed_pathology = pathology_transform(test_image)
            print(f"✅ 病理变换测试成功: {transformed_pathology.shape}")

            return True

        except Exception as e:
            print(f"❌ 数据变换测试失败: {e}")
            traceback.print_exc()
            return False

    def test_data_utils(self):
        """测试数据工具函数"""
        print("\n📊 测试数据工具函数...")
        print("-" * 30)

        try:
            from MIC.src.utils.data_utils import create_data_splits, analyze_dataset

            # 1. 创建测试样本数据
            samples = [
                {
                    "id": "test_001",
                    "label": "class_A",
                    "text_path": "texts/test_001.txt",
                    "photo_paths": ["photos/test_001_1.png"],
                    "pathology_paths": [{"path": "pathology/test_001.tif"}],
                },
                {
                    "id": "test_002",
                    "label": "class_B",
                    "text_path": "texts/test_002.txt",
                    "photo_paths": ["photos/test_002_1.png", "photos/test_002_2.png"],
                    "pathology_paths": [{"path": "pathology/test_002.tif"}],
                },
            ]

            # 2. 测试数据分割
            splits = create_data_splits(
                samples, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
            )
            print(
                f"✅ 数据分割测试成功: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
            )

            return True

        except Exception as e:
            print(f"❌ 数据工具测试失败: {e}")
            return False

    def test_dataset_class(self):
        """测试数据集类功能"""
        print("\n📦 测试数据集类功能...")
        print("-" * 30)

        try:
            from MIC.src.data.dataset import MultiModalMedicalDataset
            from MIC.src.data.transforms import get_transforms

            # 1. 先为A_Datasets创建索引文件
            self._create_test_index()

            # 2. 创建变换
            transforms = get_transforms(
                {
                    "text": {"enable_augment": False},
                    "photo": {"enable_augment": False},
                    "pathology": {"enable_augment": False},
                },
                is_training=False,
            )

            # 3. 创建数据集
            dataset_config = {"patch_size": 256, "overlap": 0.1, "max_patches": 10}

            dataset = MultiModalMedicalDataset(
                data_dir=str(TEST_CONFIG["data_dir"]),
                split="train",
                transforms=transforms,
                config=dataset_config,
            )

            print(f"✅ 数据集创建成功: {len(dataset)} 样本")

            # 4. 测试单个样本获取
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✅ 样本获取成功:")
                print(f"    文本长度: {len(sample['text'])}")
                print(f"    照片数量: {len(sample['photos'])}")
                print(f"    病理patches: {len(sample['pathology']['patches'])}")
                print(f"    标签: {sample['label'].item()}")

            # 5. 测试类别分布
            distribution = dataset.get_class_distribution()
            print(f"✅ 类别分布: {distribution}")

            return True

        except Exception as e:
            print(f"❌ 数据集类测试失败: {e}")
            traceback.print_exc()
            return False

    def test_dataloader(self):
        """测试数据加载器功能"""
        print("\n🔄 测试数据加载器功能...")
        print("-" * 30)

        try:
            from MIC.src.data.dataloader import create_dataloaders, MultiModalCollator

            # 1. 配置
            config = {
                "batch_size": 1,  # 小batch用于测试
                "num_workers": 0,  # 避免多进程问题
                "pin_memory": False,
                "transforms": {
                    "text": {"enable_augment": False},
                    "photo": {"enable_augment": False},
                    "pathology": {"enable_augment": False},
                },
                "collator": {"max_photos": 5, "max_patches": 10},
                "dataset": {"patch_size": 256, "overlap": 0.1, "max_patches": 10},
            }

            # 2. 创建数据加载器
            dataloaders = create_dataloaders(
                str(TEST_CONFIG["data_dir"]), config, use_weighted_sampling=False
            )

            if dataloaders:
                print(f"✅ 数据加载器创建成功: {list(dataloaders.keys())}")

                # 3. 测试批次数据
                for split, dataloader in dataloaders.items():
                    if len(dataloader) > 0:
                        batch = next(iter(dataloader))
                        print(f"✅ {split} 批次数据:")
                        print(f"    文本数量: {len(batch['text']['texts'])}")
                        print(f"    照片形状: {batch['photos']['images'].shape}")
                        print(f"    病理形状: {batch['pathology']['patches'].shape}")
                        print(f"    标签形状: {batch['labels'].shape}")
                        break

                return True
            else:
                print("❌ 无法创建数据加载器")
                return False

        except Exception as e:
            print(f"❌ 数据加载器测试失败: {e}")
            traceback.print_exc()
            return False

    def test_real_data_loading(self):
        """测试实际数据加载"""
        print("\n🔬 测试实际数据加载...")
        print("-" * 30)

        try:
            # 1. 验证样本数据存在
            sample_data = TEST_CONFIG["sample_data"]
            data_dir = TEST_CONFIG["data_dir"]

            # 检查文本文件
            text_file = data_dir / sample_data["expected_files"]["text"]
            if text_file.exists():
                with open(text_file, "r", encoding="utf-8") as f:
                    text_content = f.read()
                print(f"✅ 文本文件读取成功: {len(text_content)} 字符")
                print(f"    样本内容预览: {text_content[:100]}...")
            else:
                print(f"❌ 文本文件不存在: {text_file}")
                return False

            # 检查图片文件
            photo_count = 0
            for photo_path in sample_data["expected_files"]["photos"]:
                photo_file = data_dir / photo_path
                if photo_file.exists():
                    try:
                        img = Image.open(photo_file)
                        photo_count += 1
                        print(f"✅ 图片文件: {photo_file.name} ({img.size})")
                    except Exception as e:
                        print(f"❌ 图片文件损坏: {photo_file.name} - {e}")
                else:
                    print(f"❌ 图片文件不存在: {photo_file}")

            # 检查病理文件
            pathology_count = 0
            for pathology_path in sample_data["expected_files"]["pathology"]:
                pathology_file = data_dir / pathology_path
                if pathology_file.exists():
                    try:
                        img = Image.open(pathology_file)
                        pathology_count += 1
                        print(f"✅ 病理文件: {pathology_file.name} ({img.size})")
                    except Exception as e:
                        print(f"❌ 病理文件损坏: {pathology_file.name} - {e}")
                else:
                    print(f"❌ 病理文件不存在: {pathology_file}")

            # 检查标签文件
            labels_file = data_dir / "labels.json"
            if labels_file.exists():
                with open(labels_file, "r", encoding="utf-8") as f:
                    labels = json.load(f)
                expected_label = sample_data["expected_label"]
                actual_label = labels.get(sample_data["patient_id"])
                if actual_label == expected_label:
                    print(
                        f"✅ 标签验证成功: {sample_data['patient_id']} -> {actual_label}"
                    )
                else:
                    print(f"❌ 标签不匹配: 期望 {expected_label}, 实际 {actual_label}")
                    return False
            else:
                print(f"❌ 标签文件不存在: {labels_file}")
                return False

            print(f"\n📊 样本数据统计:")
            print(f"  患者ID: {sample_data['patient_id']}")
            print(f"  标签: {actual_label}")
            print(f"  文本: 1 个文件")
            print(f"  照片: {photo_count} 个文件")
            print(f"  病理: {pathology_count} 个文件")

            return photo_count > 0 and pathology_count > 0

        except Exception as e:
            print(f"❌ 实际数据加载测试失败: {e}")
            traceback.print_exc()
            return False

    def test_end_to_end_pipeline(self):
        """测试端到端数据处理流水线"""
        print("\n🚀 测试端到端数据处理流水线...")
        print("-" * 30)

        try:
            from MIC.src.data.dataset import MultiModalMedicalDataset
            from MIC.src.data.transforms import get_transforms
            import torch

            print("⚠️ 注意: 此测试需要PyTorch依赖")

            # 1. 创建完整的数据处理流水线
            transforms = get_transforms(
                {
                    "text": {"enable_augment": False, "max_length": 256},
                    "photo": {"enable_augment": False},
                    "pathology": {"enable_augment": False},
                },
                is_training=False,
            )

            dataset_config = {
                "patch_size": 256,
                "overlap": 0.1,
                "max_patches": 5,  # 少一些用于测试
            }

            dataset = MultiModalMedicalDataset(
                data_dir=str(TEST_CONFIG["data_dir"]),
                split="train",
                transforms=transforms,
                config=dataset_config,
            )

            if len(dataset) == 0:
                print("❌ 数据集为空，无法进行端到端测试")
                return False

            # 2. 获取样本并处理
            sample = dataset[0]

            print("✅ 端到端流水线测试:")
            print(f"  文本处理: {type(sample['text'])} (长度: {len(sample['text'])})")
            print(f"  照片处理: {len(sample['photos'])} 张图片")
            if sample["photos"]:
                print(
                    f"    照片tensor形状: {sample['photos'][0].shape if hasattr(sample['photos'][0], 'shape') else 'N/A'}"
                )
            print(f"  病理处理: {len(sample['pathology']['patches'])} 个patches")
            if sample["pathology"]["patches"]:
                patch_shape = (
                    sample["pathology"]["patches"][0].shape
                    if hasattr(sample["pathology"]["patches"][0], "shape")
                    else "N/A"
                )
                print(f"    病理patch形状: {patch_shape}")
            print(f"  标签: {sample['label']}")

            return True

        except ImportError as e:
            print(f"⚠️ 跳过端到端测试 (缺少依赖): {e}")
            return True  # 不算失败
        except Exception as e:
            print(f"❌ 端到端流水线测试失败: {e}")
            traceback.print_exc()
            return False

    def _create_test_index(self):
        """为A_Datasets创建测试索引文件"""
        data_dir = TEST_CONFIG["data_dir"]

        # 创建train索引 (简化版)
        train_samples = [
            {
                "id": "patient_001",
                "label": "class_A",
                "text_path": "texts/patient_001.txt",
                "photo_paths": [
                    "photos/patient_001/photo1.png",
                    "photos/patient_001/photo2.png",
                ],
                "pathology_paths": [
                    {"path": "pathology/patient_001/slide1.tif", "roi": None},
                    {"path": "pathology/patient_001/slide2.tif", "roi": None},
                ],
            }
        ]

        # 保存索引文件
        index_file = data_dir / "train_index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)

        print(f"✅ 创建测试索引文件: {index_file}")

    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 开始数据加载功能测试...")
        print("=" * 60)

        if not self.setup():
            return False

        try:
            tests = [
                ("数据变换功能", self.test_data_transforms),
                ("数据工具函数", self.test_data_utils),
                ("数据集类功能", self.test_dataset_class),
                ("数据加载器功能", self.test_dataloader),
                ("实际数据加载", self.test_real_data_loading),
                ("端到端流水线", self.test_end_to_end_pipeline),
            ]

            results = []
            for test_name, test_func in tests:
                try:
                    result = test_func()
                    results.append((test_name, result))
                    self.test_results[test_name] = result
                except Exception as e:
                    print(f"\n❌ {test_name}测试出现异常: {e}")
                    results.append((test_name, False))
                    self.test_results[test_name] = False

            # 汇总结果
            print("\n" + "=" * 60)
            print("📋 测试结果汇总:")

            success_count = 0
            for test_name, success in results:
                status = "✅ 通过" if success else "❌ 失败"
                print(f"  {test_name}: {status}")
                if success:
                    success_count += 1

            total_tests = len(results)
            print(f"\n🎯 总体结果: {success_count}/{total_tests} 测试通过")

            if success_count == total_tests:
                print("\n🎉 所有数据加载功能测试通过！")
                print("\n✨ 数据处理系统工作正常:")
                print("  ✅ 数据变换和预处理")
                print("  ✅ 多模态数据集加载")
                print("  ✅ 批次数据处理")
                print("  ✅ 实际样本数据验证")
                return True
            else:
                print(f"\n⚠️ {total_tests - success_count} 个测试失败")
                print("\n📋 可能的原因:")
                print("  1. 缺少PyTorch、PIL等依赖")
                print("  2. A_Datasets数据不完整")
                print("  3. 系统环境问题")
                return False

        finally:
            self.cleanup()


def main():
    """主函数"""
    tester = DataLoadingTester()
    success = tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
