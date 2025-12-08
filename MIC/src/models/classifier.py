"""
分类器模块
实现多层感知机分类器和各种分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class MLPClassifier(nn.Module):
    """
    多层感知机分类器
    标准的深度神经网络分类器
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout

        # 激活函数选择
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()

        # 构建网络层
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # 线性层
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # 批归一化
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # 激活函数
            layers.append(self.activation)

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features: [batch_size, input_dim]

        Returns:
            logits: [batch_size, num_classes]
        """
        return self.classifier(features)


class AttentionClassifier(nn.Module):
    """
    基于注意力机制的分类器
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_heads: int = 8,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim or input_dim

        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # 特征变换
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, input_dim),
        )

        # 分类头
        self.classifier_head = nn.Linear(input_dim, num_classes)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            features: [batch_size, input_dim]

        Returns:
            包含logits和注意力权重的字典
        """
        # 添加序列维度用于注意力计算
        features_seq = features.unsqueeze(1)  # [batch_size, 1, input_dim]

        # 自注意力
        attended_features, attention_weights = self.self_attention(
            features_seq, features_seq, features_seq
        )

        # 残差连接和layer norm
        attended_features = self.layer_norm(features_seq + attended_features)

        # 特征变换
        transformed_features = self.feature_transform(attended_features.squeeze(1))

        # 分类
        logits = self.classifier_head(transformed_features)

        return {
            "logits": logits,
            "attention_weights": attention_weights.squeeze(),
            "features": transformed_features,
        }


class MultiTaskClassifier(nn.Module):
    """
    多任务分类器
    支持主任务和辅助任务的联合学习
    """

    def __init__(
        self,
        input_dim: int,
        main_num_classes: int,
        aux_tasks: Optional[Dict[str, int]] = None,
        shared_hidden_dims: Optional[List[int]] = None,
        task_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.main_num_classes = main_num_classes
        self.aux_tasks = aux_tasks or {}

        # 默认隐藏层配置
        shared_hidden_dims = shared_hidden_dims or [256, 128]
        task_hidden_dims = task_hidden_dims or [64]

        # 共享特征提取器
        self.shared_encoder = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=shared_hidden_dims,
            num_classes=shared_hidden_dims[-1],  # 输出特征而非分类
            dropout=dropout,
            use_batch_norm=True,
        )

        # 移除共享编码器的最后一层（分类层）
        self.shared_encoder.classifier = self.shared_encoder.classifier[:-1]

        # 主任务分类头
        self.main_classifier = MLPClassifier(
            input_dim=shared_hidden_dims[-1],
            hidden_dims=task_hidden_dims,
            num_classes=main_num_classes,
            dropout=dropout,
        )

        # 辅助任务分类头
        self.aux_classifiers = nn.ModuleDict()
        for task_name, num_classes in self.aux_tasks.items():
            self.aux_classifiers[task_name] = MLPClassifier(
                input_dim=shared_hidden_dims[-1],
                hidden_dims=task_hidden_dims,
                num_classes=num_classes,
                dropout=dropout,
            )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            features: [batch_size, input_dim]

        Returns:
            包含各任务logits的字典
        """
        # 共享特征提取
        shared_features = self.shared_encoder(features)

        # 主任务预测
        main_logits = self.main_classifier(shared_features)

        result = {"main_logits": main_logits, "shared_features": shared_features}

        # 辅助任务预测
        for task_name, classifier in self.aux_classifiers.items():
            aux_logits = classifier(shared_features)
            result[f"{task_name}_logits"] = aux_logits

        return result


class UncertaintyAwareClassifier(nn.Module):
    """
    不确定性感知分类器
    输出预测的不确定性估计
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
        uncertainty_type: str = "epistemic",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.uncertainty_type = uncertainty_type  # 'epistemic' or 'aleatoric'

        # 主分类器
        self.main_classifier = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )

        if uncertainty_type == "aleatoric":
            # 随机不确定性：预测方差
            self.uncertainty_head = MLPClassifier(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,  # 每个类别一个方差
                dropout=dropout,
            )
        elif uncertainty_type == "epistemic":
            # 认知不确定性：使用MC Dropout
            self.dropout_layers = nn.ModuleList(
                [nn.Dropout(dropout) for _ in range(len(hidden_dims))]
            )

    def forward(
        self, features: torch.Tensor, num_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            features: [batch_size, input_dim]
            num_samples: MC采样次数（仅用于epistemic uncertainty）

        Returns:
            包含logits和不确定性的字典
        """
        if self.uncertainty_type == "aleatoric":
            # 随机不确定性
            logits = self.main_classifier(features)
            log_var = self.uncertainty_head(features)

            return {
                "logits": logits,
                "log_variance": log_var,
                "variance": torch.exp(log_var),
            }

        elif self.uncertainty_type == "epistemic":
            # 认知不确定性：MC Dropout
            if self.training:
                # 训练时正常前向传播
                logits = self.main_classifier(features)
                return {"logits": logits}
            else:
                # 推理时进行多次采样
                self.train()  # 启用dropout
                predictions = []

                for _ in range(num_samples):
                    logits = self.main_classifier(features)
                    predictions.append(F.softmax(logits, dim=1))

                self.eval()  # 恢复eval模式

                # 计算统计量
                predictions = torch.stack(
                    predictions
                )  # [num_samples, batch_size, num_classes]
                mean_pred = predictions.mean(dim=0)
                variance = predictions.var(dim=0)

                return {
                    "logits": torch.log(mean_pred + 1e-8),  # 转换回logits
                    "mean_prediction": mean_pred,
                    "variance": variance,
                    "entropy": -torch.sum(
                        mean_pred * torch.log(mean_pred + 1e-8), dim=1
                    ),
                }


def create_classifier(config: Dict) -> nn.Module:
    """
    创建分类器工厂函数

    Args:
        config: 分类器配置字典

    Returns:
        分类器实例
    """
    classifier_type = config.get("type", "mlp")
    input_dim = config["input_dim"]
    num_classes = config["num_classes"]

    if classifier_type == "mlp":
        return MLPClassifier(
            input_dim=input_dim,
            hidden_dims=config.get("hidden_dims", [256, 128]),
            num_classes=num_classes,
            dropout=config.get("dropout", 0.3),
            activation=config.get("activation", "relu"),
            use_batch_norm=config.get("use_batch_norm", True),
        )

    elif classifier_type == "attention":
        return AttentionClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            num_heads=config.get("num_heads", 8),
            hidden_dim=config.get("hidden_dim", None),
            dropout=config.get("dropout", 0.3),
        )

    elif classifier_type == "multitask":
        return MultiTaskClassifier(
            input_dim=input_dim,
            main_num_classes=num_classes,
            aux_tasks=config.get("aux_tasks", {}),
            shared_hidden_dims=config.get("shared_hidden_dims", None),
            task_hidden_dims=config.get("task_hidden_dims", None),
            dropout=config.get("dropout", 0.3),
        )

    elif classifier_type == "uncertainty":
        return UncertaintyAwareClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=config.get("hidden_dims", [256, 128]),
            dropout=config.get("dropout", 0.3),
            uncertainty_type=config.get("uncertainty_type", "epistemic"),
        )

    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")


if __name__ == "__main__":
    # 测试各种分类器
    configs = [
        {
            "type": "mlp",
            "input_dim": 512,
            "num_classes": 6,
            "hidden_dims": [256, 128],
            "dropout": 0.3,
        },
        {
            "type": "attention",
            "input_dim": 512,
            "num_classes": 6,
            "num_heads": 8,
            "dropout": 0.3,
        },
        {
            "type": "multitask",
            "input_dim": 512,
            "num_classes": 6,
            "aux_tasks": {"severity": 3, "location": 4},
            "dropout": 0.3,
        },
        {
            "type": "uncertainty",
            "input_dim": 512,
            "num_classes": 6,
            "hidden_dims": [256, 128],
            "uncertainty_type": "epistemic",
        },
    ]

    # 测试数据
    batch_size = 4
    test_features = torch.randn(batch_size, 512)

    # 测试每种分类器
    for i, config in enumerate(configs):
        print(f"\n测试分类器 {i+1}: {config['type']}")

        classifier = create_classifier(config)

        if config["type"] == "mlp":
            logits = classifier(test_features)
            print(f"  输出形状: {logits.shape}")

        elif config["type"] == "attention":
            result = classifier(test_features)
            print(f"  logits形状: {result['logits'].shape}")
            print(f"  attention权重形状: {result['attention_weights'].shape}")

        elif config["type"] == "multitask":
            result = classifier(test_features)
            print(f"  主任务logits形状: {result['main_logits'].shape}")
            for task_name in config["aux_tasks"]:
                print(
                    f"  {task_name} logits形状: {result[f'{task_name}_logits'].shape}"
                )

        elif config["type"] == "uncertainty":
            classifier.eval()
            result = classifier(test_features, num_samples=5)
            print(f"  logits形状: {result['logits'].shape}")
            if "variance" in result:
                print(f"  方差形状: {result['variance'].shape}")

    print("\n分类器测试完成!")
