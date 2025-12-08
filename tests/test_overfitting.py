"""
过拟合测试 (Overfitting Test / Sanity Check)
验证模型是否有能力学习和记忆单条数据
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from PIL import Image
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_single_sample_overfitting():
    """
    测试模型能否过拟合单条训练数据

    预期结果：
    1. 训练损失应该降到接近0
    2. 训练准确率应该达到100%
    3. 推理结果应该与训练标签完全一致
    """
    print("=" * 80)
    print("🧪 过拟合测试 - 验证模型能否记忆单条数据")
    print("=" * 80)

    try:
        from MIC.src.data.dataset import MultiModalMedicalDataset
        from MIC.src.data.transforms import (
            PhotoTransforms,
            PathologyTransforms,
            TextTransforms,
        )
        from MIC.src.data.dataloader import MultiModalCollator
        from MIC.src.models.multimodal_model import MultiModalMedicalClassifier
        from MIC.src.training.loss import create_loss_function

        print("\n✅ 模块导入成功")
    except Exception as e:
        print(f"\n❌ 模块导入失败: {e}")
        return

    # 准备测试数据
    print("\n" + "=" * 80)
    print("📊 准备单条测试数据")
    print("=" * 80)

    data_dir = project_root / "A_Datasets"

    # 创建单样本索引（注意：索引文件应该是样本列表，不是包含"samples"键的字典）
    patient_id = "patient_001"
    single_sample_index = [
        {
            "id": patient_id,
            "text_path": "texts/patient_001.txt",
            "photo_paths": [
                "photos/patient_001/photo1.png",
                "photos/patient_001/photo2.png",
            ],
            "pathology_paths": [
                "pathology/patient_001/slide1.tif",
                "pathology/patient_001/slide2.tif",
            ],
            "label": "class_A",
        }
    ]

    # 保存临时索引文件
    train_index_path = data_dir / "overfit_train_index.json"
    with open(train_index_path, "w", encoding="utf-8") as f:
        json.dump(single_sample_index, f, indent=2, ensure_ascii=False)

    print(f"✅ 创建单样本训练集")
    print(f"  样本ID: {patient_id}")
    print(f"  标签: class_A")

    # 创建数据集
    try:
        transforms = {
            "text": TextTransforms(),
            "photo": PhotoTransforms(is_training=False),  # 不使用数据增强，便于复现
            "pathology": PathologyTransforms(is_training=False),
        }

        config = {
            "patch_size": 256,
            "overlap": 0.1,
            "max_patches": 50,  # 减少patches数量，加快训练
        }

        train_dataset = MultiModalMedicalDataset(
            data_dir=str(data_dir),
            split="overfit_train",
            transforms=transforms,
            config=config,
        )

        print(f"\n✅ 数据集创建成功")
        print(f"  样本数量: {len(train_dataset)}")
        print(f"  标签映射: {train_dataset.label_map}")

    except Exception as e:
        print(f"\n❌ 数据集创建失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 创建DataLoader
    collator = MultiModalCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,  # 不打乱，便于观察
        collate_fn=collator,
    )

    # 创建模型
    print("\n" + "=" * 80)
    print("🏗️  创建模型")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    model_config = {
        "num_classes": 6,
        "use_auxiliary_loss": False,  # 单样本训练必须关闭（辅助分类器使用BatchNorm）
        "modal_dropout_prob": 0.0,  # 单样本训练也关闭modal dropout
        # 文本编码器配置
        "text_encoder": {
            "model_name": "/Users/frozen/PycharmProjects/Qwen3-Embedding-0.6B",
            "embedding_dim": 256,
            "max_length": 256,
            "freeze_encoder": False,
        },
        # 照片编码器配置
        "photo_encoder": {
            "backbone": "/Users/frozen/PycharmProjects/vit-base-patch16-224",
            "pretrained": True,
            "feature_dim": 256,
            "num_heads": 4,
            "dropout": 0.1,
            "max_images": 5,
        },
        # 病理编码器配置
        "pathology_encoder": {
            "patch_size": 224,
            "overlap": 0.1,
            "patch_backbone": "/Users/frozen/PycharmProjects/vit-base-patch16-224",
            "patch_feature_dim": 128,
            "mil_feature_dim": 256,
            "attention_heads": 4,
            "dropout": 0.1,
            "max_patches": 30,
        },
        # 融合模块配置
        "fusion": {
            "fusion_dim": 256,
            "attention_heads": 4,
            "dropout": 0.2,
            "num_layers": 2,
            "fusion_strategy": "transformer",
        },
        # 分类器配置
        "classifier": {
            "type": "mlp",
            "hidden_dims": [128, 64],
            "dropout": 0.3,
            "activation": "relu",
            "use_batch_norm": False,  # 数据少时关闭BatchNorm
        },
    }

    try:
        model = MultiModalMedicalClassifier(model_config)
        model = model.to(device)
        print("✅ 模型创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 较大学习率，加快收敛

    loss_config = {
        "type": "multimodal",
        "num_classes": model_config["num_classes"],  # 添加 num_classes 到配置中
        "main_loss_type": "cross_entropy",
        "loss_weights": {
            "main_loss": 1.0,
            "text_aux": 0.0,  # 关闭辅助损失，专注主任务
            "photo_aux": 0.0,
            "pathology_aux": 0.0,
        },
    }
    criterion = create_loss_function(loss_config)

    # 训练循环
    print("\n" + "=" * 80)
    print("🚀 开始过拟合训练")
    print("=" * 80)

    num_epochs = 10  # 训练足够多的轮次
    target_loss = 0.01  # 目标损失

    model.train()
    losses = []

    print(f"训练参数:")
    print(f"  - 轮次: {num_epochs}")
    print(f"  - 学习率: 1e-3")
    print(f"  - 目标损失: {target_loss}")
    print(f"  - 期望: 损失应降到接近0，准确率应达到100%")
    print()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            # 移动数据到设备
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            targets = batch["labels"]

            # 前向传播
            optimizer.zero_grad()
            outputs = model(batch)

            # 计算损失
            modal_availability = batch.get(
                "modal_availability", torch.ones(len(targets), 3, device=device)
            )
            loss_dict = criterion(outputs, targets, modal_availability)
            loss = loss_dict["total_loss"]

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            epoch_loss += loss.item()
            predictions = outputs["predictions"].argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += len(targets)

        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total * 100
        losses.append(avg_loss)

        print(
            f"Epoch {epoch+1:3d}/{num_epochs}: Loss = {avg_loss:.6f}, Accuracy = {accuracy:.1f}%"
        )

        # 如果达到目标损失，提前停止
        if avg_loss < target_loss:
            print(f"\n🎉 在第 {epoch+1} 轮达到目标损失")

    # 推理测试
    print("\n" + "=" * 80)
    print("🔍 推理测试 - 验证模型是否记住了训练数据")
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            # 移动数据到设备
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            targets = batch["labels"]

            # 推理
            outputs = model(batch)
            predictions = outputs["predictions"]
            pred_probs = torch.softmax(predictions, dim=1)
            pred_class = predictions.argmax(dim=1)

            # 输出结果
            true_label = targets[0].item()
            pred_label = pred_class[0].item()
            confidence = pred_probs[0, pred_label].item()

            print(f"\n推理结果:")
            print(f"  真实标签: {true_label} (class_A)")
            print(f"  预测标签: {pred_label}")
            print(f"  预测概率: {pred_probs[0].cpu().numpy()}")
            print(f"  置信度: {confidence:.4f}")
            print(f"  预测正确: {'✅ 是' if pred_label == true_label else '❌ 否'}")

    # 结果分析
    print("\n" + "=" * 80)
    print("📊 结果分析")
    print("=" * 80)

    final_loss = losses[-1]
    passed = final_loss < 0.1 and pred_label == true_label

    print(f"\n训练过程:")
    print(f"  初始损失: {losses[0]:.6f}")
    print(f"  最终损失: {final_loss:.6f}")
    print(f"  损失下降: {(1 - final_loss/losses[0]) * 100:.1f}%")

    print(f"\n推理结果:")
    print(f"  预测正确: {'✅' if pred_label == true_label else '❌'}")
    print(f"  置信度: {confidence:.4f}")

    print(f"\n判定标准:")
    print(
        f"  1. 最终损失 < 0.1: {'✅ 通过' if final_loss < 0.1 else '❌ 失败'} (实际: {final_loss:.6f})"
    )
    print(f"  2. 预测正确: {'✅ 通过' if pred_label == true_label else '❌ 失败'}")
    print(
        f"  3. 置信度 > 0.9: {'✅ 通过' if confidence > 0.9 else '⚠️  警告'} (实际: {confidence:.4f})"
    )

    if passed:
        print(f"\n{'='*80}")
        print("🎉 过拟合测试通过！")
        print("=" * 80)
        print("\n✅ 模型能够成功学习和记忆单条数据")
        print("✅ 模型架构、损失函数、优化器实现正确")
        print("✅ 可以开始完整训练了")
    else:
        print(f"\n{'='*80}")
        print("❌ 过拟合测试失败！")
        print("{'='*80}")
        print("\n可能的原因:")
        if final_loss >= 0.1:
            print("  ❌ 损失没有充分下降")
            print("     - 检查学习率是否合适")
            print("     - 检查梯度是否正常传播")
            print("     - 检查损失函数实现")
        if pred_label != true_label:
            print("  ❌ 预测错误")
            print("     - 检查模型输出维度")
            print("     - 检查标签映射是否正确")
            print("     - 检查损失函数与标签格式匹配")

    # 清理
    print(f"\n🗑️  清理临时文件...")
    train_index_path.unlink()
    print("✅ 清理完成")

    return passed


if __name__ == "__main__":
    print("\n" + "🔬" * 40)
    print("过拟合测试 (Overfitting Test)")
    print("目的: 验证模型是否能够学习和记忆单条数据")
    print("🔬" * 40 + "\n")

    try:
        success = test_single_sample_overfitting()

        print("\n" + "=" * 80)
        print("测试完成")
        print("=" * 80)

        if success:
            print("\n✅ 所有测试通过！模型实现正确。")
            sys.exit(0)
        else:
            print("\n❌ 测试失败！请检查模型实现。")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ 测试过程中出现异常: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
