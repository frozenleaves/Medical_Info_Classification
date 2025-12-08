#!/usr/bin/env python3
"""
测试重构后的代码结构是否正确
"""

import sys
from pathlib import Path
import traceback


def test_imports():
    """测试所有重要模块的导入"""
    print("🔍 测试模块导入...")

    tests = [
        # 测试MIC包导入
        ("MIC包主导入", "from MIC import create_multimodal_model, MultiModalTrainer"),
        # 测试各个子模块
        ("配置模块", "from MIC.src.config import ModelConfig, TrainingConfig"),
        (
            "数据模块",
            "from MIC.src.data import MultiModalMedicalDataset, create_dataloaders",
        ),
        ("模型模块", "from MIC.src.models import create_multimodal_model"),
        (
            "编码器模块",
            "from MIC.src.models.encoders import create_text_encoder, create_photo_encoder, create_pathology_encoder",
        ),
        ("融合模块", "from MIC.src.models.fusion import create_fusion_module"),
        (
            "训练模块",
            "from MIC.src.training import MultiModalTrainer, create_loss_function",
        ),
        ("工具模块", "from MIC.src.utils import setup_logging, set_random_seed"),
    ]

    success_count = 0
    total_count = len(tests)

    for test_name, import_statement in tests:
        try:
            exec(import_statement)
            print(f"  ✅ {test_name}: 导入成功")
            success_count += 1
        except Exception as e:
            print(f"  ❌ {test_name}: 导入失败")
            print(f"     错误: {e}")

    print(f"\n📊 导入测试结果: {success_count}/{total_count} 成功")
    return success_count == total_count


def test_model_creation():
    """测试模型创建"""
    print("\n🏗️ 测试模型创建...")

    try:
        from MIC.src.models import create_multimodal_model

        # 简化配置用于测试
        config = {
            "num_classes": 6,
            "text_encoder": {
                "model_name": "bert-base-chinese",
                "embedding_dim": 128,  # 小一些用于测试
                "max_length": 128,
            },
            "photo_encoder": {
                "backbone": "resnet18",  # 轻量级模型用于测试
                "feature_dim": 128,
                "num_heads": 4,
                "max_images": 3,
            },
            "pathology_encoder": {
                "patch_backbone": "resnet18",
                "patch_feature_dim": 128,
                "mil_feature_dim": 128,
                "max_patches": 10,
            },
            "fusion": {"fusion_dim": 128, "attention_heads": 4, "num_layers": 1},
            "classifier": {"type": "mlp", "hidden_dims": [64], "dropout": 0.1},
        }

        model = create_multimodal_model(config)
        print("  ✅ 模型创建成功")

        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  📊 模型参数数量: {total_params:,}")

        return True

    except Exception as e:
        print(f"  ❌ 模型创建失败: {e}")
        traceback.print_exc()
        return False


def test_data_processing():
    """测试数据处理模块"""
    print("\n📊 测试数据处理...")

    try:
        from MIC.src.data.transforms import get_transforms
        from MIC.src.utils.data_utils import analyze_dataset

        # 测试数据变换
        config = {
            "text": {"enable_augment": False},
            "photo": {"enable_augment": False},
            "pathology": {"enable_augment": False},
        }

        transforms = get_transforms(config, is_training=False)
        print("  ✅ 数据变换创建成功")

        # 测试数据集分析（如果A_Datasets存在）
        a_datasets_path = Path("A_Datasets")
        if a_datasets_path.exists():
            print("  🔍 找到测试数据集，进行分析...")
            # 这里可以添加更详细的数据集测试
            print("  ✅ 数据集路径检查通过")
        else:
            print("  ⚠️ 未找到A_Datasets目录，跳过数据集分析")

        return True

    except Exception as e:
        print(f"  ❌ 数据处理测试失败: {e}")
        return False


def test_training_components():
    """测试训练相关组件"""
    print("\n🎯 测试训练组件...")

    try:
        from MIC.src.training.loss import create_loss_function
        from MIC.src.training.metrics import MultiModalMetrics

        # 测试损失函数创建
        loss_config = {"type": "multimodal", "num_classes": 6}

        loss_fn = create_loss_function(loss_config)
        print("  ✅ 损失函数创建成功")

        # 测试评估指标
        metrics = MultiModalMetrics(6)
        print("  ✅ 评估指标创建成功")

        return True

    except Exception as e:
        print(f"  ❌ 训练组件测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始测试代码结构...")
    print("=" * 60)

    # 添加项目根目录到Python路径
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    # 运行各项测试
    tests = [
        ("模块导入", test_imports),
        ("模型创建", test_model_creation),
        ("数据处理", test_data_processing),
        ("训练组件", test_training_components),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name}测试出现异常: {e}")
            results.append((test_name, False))

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
        print("🎉 所有测试通过！代码结构重构成功！\n\n")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查相关问题")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
