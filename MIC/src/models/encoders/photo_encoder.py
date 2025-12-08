"""
口腔照片编码器模块
使用ViT进行特征提取，结合多图片注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np


class MultiImageAttention(nn.Module):
    """
    多图片注意力机制
    用于从多张口腔照片中提取最相关的特征
    """

    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert (
            self.head_dim * num_heads == feature_dim
        ), "feature_dim必须能被num_heads整除"

        # 查询、键、值投影层
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        # 输出投影
        self.output_proj = nn.Linear(feature_dim, feature_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 缩放因子
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self, image_features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            image_features: [batch_size, num_images, feature_dim]
            mask: [batch_size, num_images] 有效图片掩码

        Returns:
            aggregated_features: [batch_size, feature_dim]
            attention_weights: [batch_size, num_images]
        """
        batch_size, num_images, feature_dim = image_features.shape

        # 投影到QKV
        Q = self.query_proj(image_features)  # [batch_size, num_images, feature_dim]
        K = self.key_proj(image_features)
        V = self.value_proj(image_features)

        # 重塑为多头形式
        Q = Q.view(batch_size, num_images, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        K = K.view(batch_size, num_images, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        V = V.view(batch_size, num_images, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        # 现在形状: [batch_size, num_heads, num_images, head_dim]

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [batch_size, num_heads, num_images, num_images]

        # 应用掩码
        if mask is not None:
            # 扩展mask维度以匹配attention_scores
            expanded_mask = mask.unsqueeze(1).unsqueeze(
                2
            )  # [batch_size, 1, 1, num_images]
            expanded_mask = expanded_mask.expand(-1, self.num_heads, num_images, -1)
            attention_scores = attention_scores.masked_fill(
                ~expanded_mask, float("-inf")
            )

        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        attended_features = torch.matmul(attention_weights, V)
        # [batch_size, num_heads, num_images, head_dim]

        # 合并多头
        attended_features = (
            attended_features.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_images, feature_dim)
        )

        # 输出投影
        output_features = self.output_proj(attended_features)

        # 全局聚合：加权平均
        if mask is not None:
            # 计算每张图片的重要性权重
            global_weights = attention_weights.mean(dim=1).mean(
                dim=1
            )  # [batch_size, num_images]
            global_weights = global_weights.masked_fill(~mask, 0)
            global_weights = global_weights / (
                global_weights.sum(dim=1, keepdim=True) + 1e-8
            )
        else:
            global_weights = (
                torch.ones(batch_size, num_images, device=image_features.device)
                / num_images
            )

        # 加权聚合
        aggregated_features = torch.sum(
            output_features * global_weights.unsqueeze(-1), dim=1
        )  # [batch_size, feature_dim]

        return aggregated_features, global_weights


class PhotoEncoder(nn.Module):
    """
    照片编码器
    使用ViT backbone + 多图片注意力机制
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # 基本配置
        self.backbone_name = config.get("backbone", "vit_base_patch16_224")
        self.pretrained = config.get("pretrained", True)
        self.feature_dim = config.get("feature_dim", 768)
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        self.max_images = config.get("max_images", 10)

        # 加载ViT backbone
        self.backbone = self._load_backbone()

        # 特征投影层（如果需要）
        backbone_dim = self.backbone.num_features
        if backbone_dim != self.feature_dim:
            self.feature_projection = nn.Linear(backbone_dim, self.feature_dim)
        else:
            self.feature_projection = nn.Identity()

        # 多图片注意力机制
        self.multi_image_attention = MultiImageAttention(
            feature_dim=self.feature_dim, num_heads=self.num_heads, dropout=self.dropout
        )

        # 特征增强层
        self.feature_enhancement = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.Dropout(self.dropout),
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(self.feature_dim)

        print(f"口腔照片编码器初始化完成:")
        print(f"  骨干网络: {self.backbone_name}")
        print(f"  特征维度: {self.feature_dim}")
        print(f"  注意力头数: {self.num_heads}")
        print(f"  最大图片数: {self.max_images}")

    def _load_backbone(self) -> nn.Module:
        """加载ViT backbone"""
        from pathlib import Path

        # 检查是否是本地路径
        if isinstance(self.backbone_name, str) and (
            self.backbone_name.startswith("/")
            or self.backbone_name.startswith("./")
            or Path(self.backbone_name).exists()
        ):
            # 本地路径处理
            return self._load_local_backbone()
        else:
            # 使用timm加载预训练模型
            return self._load_timm_backbone()

    def _load_local_backbone(self) -> nn.Module:
        """加载本地模型"""
        from pathlib import Path
        import warnings

        model_path = Path(self.backbone_name)

        try:
            print(f"尝试从本地路径加载模型: {model_path}")

            # 方法1: 尝试使用transformers库加载
            try:
                from transformers import AutoModel, AutoImageProcessor

                model = AutoModel.from_pretrained(
                    str(model_path), trust_remote_code=True
                )

                # 添加timm模型兼容属性
                if hasattr(model.config, "hidden_size"):
                    model.num_features = model.config.hidden_size
                elif hasattr(model, "embed_dim"):
                    model.num_features = model.embed_dim
                else:
                    model.num_features = 768  # 默认值

                # 添加forward_features方法以兼容timm接口
                if not hasattr(model, "forward_features"):

                    def forward_features(x):
                        outputs = model(pixel_values=x)
                        if hasattr(outputs, "last_hidden_state"):
                            return outputs.last_hidden_state
                        elif hasattr(outputs, "pooler_output"):
                            return outputs.pooler_output
                        else:
                            return outputs[0] if isinstance(outputs, tuple) else outputs

                    model.forward_features = forward_features

                print(
                    f"成功使用transformers加载本地模型, 特征维度: {model.num_features}"
                )
                return model

            except ImportError:
                print("transformers库未安装，尝试其他方法...")
            except Exception as e:
                print(f"transformers加载失败: {e}")

            # 方法2: 尝试直接加载PyTorch模型
            if (model_path / "pytorch_model.bin").exists() or (
                model_path / "model.safetensors"
            ).exists():
                try:
                    # 尝试加载配置
                    config_path = model_path / "config.json"
                    if config_path.exists():
                        import json

                        with open(config_path, "r") as f:
                            config = json.load(f)

                        # 根据配置创建模型架构
                        model_type = config.get("model_type", "vit")
                        if "vit" in model_type.lower():
                            # 使用timm创建相似的架构
                            model = timm.create_model(
                                "vit_base_patch16_224",
                                pretrained=False,
                                num_classes=0,
                                global_pool="",
                            )

                            # 尝试加载权重
                            state_dict_path = model_path / "pytorch_model.bin"
                            if state_dict_path.exists():
                                state_dict = torch.load(
                                    state_dict_path, map_location="cpu"
                                )
                                model.load_state_dict(state_dict, strict=False)

                            print(
                                f"成功从本地加载ViT模型, 特征维度: {model.num_features}"
                            )
                            return model

                except Exception as e:
                    print(f"PyTorch模型加载失败: {e}")

            # 方法3: 如果本地路径加载失败，尝试从路径名推断模型类型
            path_name = model_path.name.lower()
            if "vit" in path_name:
                # 尝试匹配ViT模型
                if "base" in path_name:
                    model_name = "vit_base_patch16_224"
                elif "large" in path_name:
                    model_name = "vit_large_patch16_224"
                elif "small" in path_name:
                    model_name = "vit_small_patch16_224"
                else:
                    model_name = "vit_base_patch16_224"

                print(f"根据路径名推断模型类型，使用 {model_name}")
                model = timm.create_model(
                    model_name,
                    pretrained=self.pretrained,
                    num_classes=0,
                    global_pool="",
                )
                return model

        except Exception as e:
            print(f"本地模型加载失败: {e}")

        # 最后的回退方案
        print("回退到默认的vit_base_patch16_224模型")
        return self._load_timm_backbone("vit_base_patch16_224")

    def _load_timm_backbone(self, model_name: str = None) -> nn.Module:
        """使用timm加载模型"""
        model_name = model_name or self.backbone_name

        try:
            model = timm.create_model(
                model_name,
                pretrained=self.pretrained,
                num_classes=0,  # 移除分类头
                global_pool="",  # 移除全局池化
            )

            print(f"成功加载 {model_name}, 特征维度: {model.num_features}")
            return model

        except Exception as e:
            print(f"警告: 无法加载 {model_name}: {e}")
            print("回退到 vit_base_patch16_224")

            # 回退方案
            model = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0, global_pool=""
            )
            return model

    def forward(
        self, images: torch.Tensor, masks: torch.Tensor, return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            images: [batch_size, max_images, 3, H, W]
            masks: [batch_size, max_images] 有效图片掩码
            return_attention: 是否返回注意力权重

        Returns:
            包含编码特征的字典
        """
        batch_size, max_images, channels, height, width = images.shape

        # 将batch和image维度合并进行特征提取
        images_flat = images.view(batch_size * max_images, channels, height, width)

        # 提取单张图片特征
        with torch.amp.autocast("cuda", enabled=False):  # 某些ViT对mixed precision敏感
            backbone_features = self.backbone.forward_features(images_flat)

            # 智能处理不同类型backbone的输出
            if len(backbone_features.shape) == 4:
                # CNN类型（ResNet, EfficientNet等）: [batch, channels, height, width]
                # 需要全局平均池化将空间维度压缩
                backbone_features = F.adaptive_avg_pool2d(backbone_features, (1, 1))
                backbone_features = backbone_features.flatten(1)  # [batch, channels]

            elif len(backbone_features.shape) == 3:
                # Transformer类型（ViT等）: [batch, num_tokens, embed_dim]
                # 例如: [batch, 197, 768] (1个CLS + 196个patch tokens)
                #
                # 使用CLS token作为全局特征的原因:
                # 1. CLS token通过多层自注意力聚合了所有patch的语义信息
                # 2. 预训练时已针对分类任务优化，包含最相关的全局表示
                # 3. 相比简单平均，CLS token包含学习到的自适应注意力权重
                # 4. ViT论文验证这是性能最优的方案
                # 5. 信息压缩：从150K+维 → 768维，但保留关键语义
                #
                # 参考: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
                backbone_features = backbone_features[:, 0]  # 取第一个token (CLS)

            elif len(backbone_features.shape) == 2:
                # 已经是向量形式: [batch, features]
                # 无需处理，直接使用
                pass

            else:
                raise ValueError(
                    f"不支持的backbone输出形状: {backbone_features.shape}。"
                    f"期望2D(向量)、3D(Transformer)或4D(CNN)张量。"
                )

        # 特征投影
        image_features = self.feature_projection(backbone_features)
        image_features = self.layer_norm(image_features)

        # 重塑回原始维度
        image_features = image_features.view(batch_size, max_images, self.feature_dim)

        # 应用多图片注意力
        aggregated_features, attention_weights = self.multi_image_attention(
            image_features, masks
        )

        # 特征增强
        enhanced_features = self.feature_enhancement(aggregated_features)

        # 残差连接
        final_features = aggregated_features + enhanced_features
        final_features = self.layer_norm(final_features)

        result = {
            "features": final_features,  # [batch_size, feature_dim]
            "individual_features": image_features,  # [batch_size, max_images, feature_dim]
            "masks": masks,
        }

        if return_attention:
            result["attention_weights"] = attention_weights

        return result

    def encode_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码单张图片（用于推理）"""
        self.eval()
        with torch.no_grad():
            # 添加batch和image维度
            if len(image.shape) == 3:
                image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
            elif len(image.shape) == 4:
                image = image.unsqueeze(1)  # [B, 1, 3, H, W]

            # 创建掩码
            batch_size = image.shape[0]
            mask = torch.ones(batch_size, 1, dtype=torch.bool, device=image.device)

            # 前向传播
            result = self.forward(image, mask)
            return result["features"]

    def encode_multiple_images(self, images_list: List[torch.Tensor]) -> torch.Tensor:
        """编码多张图片列表（用于推理）"""
        self.eval()
        with torch.no_grad():
            batch_results = []

            for images in images_list:
                if isinstance(images, list):
                    # 将图片列表转换为tensor
                    max_len = min(len(images), self.max_images)
                    if max_len == 0:
                        # 空图片列表，创建零特征
                        features = torch.zeros(
                            1, self.feature_dim, device=next(self.parameters()).device
                        )
                    else:
                        # 堆叠图片
                        images_tensor = torch.stack(images[:max_len])

                        # 补齐到max_images
                        if images_tensor.shape[0] < self.max_images:
                            padding_size = self.max_images - images_tensor.shape[0]
                            padding = torch.zeros(
                                padding_size,
                                *images_tensor.shape[1:],
                                device=images_tensor.device,
                            )
                            images_tensor = torch.cat([images_tensor, padding], dim=0)

                        # 添加batch维度
                        images_tensor = images_tensor.unsqueeze(
                            0
                        )  # [1, max_images, 3, H, W]

                        # 创建掩码
                        mask = torch.zeros(
                            1,
                            self.max_images,
                            dtype=torch.bool,
                            device=images_tensor.device,
                        )
                        mask[0, :max_len] = True

                        # 前向传播
                        result = self.forward(images_tensor, mask)
                        features = result["features"]

                batch_results.append(features)

            return torch.cat(batch_results, dim=0)


class MedicalPhotoEncoder(PhotoEncoder):
    """
    医学照片编码器
    针对医学图像的特殊优化
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        # 医学图像特定的预处理
        self.medical_preprocessing = config.get("medical_preprocessing", True)

        if self.medical_preprocessing:
            # 添加医学图像特定的特征增强
            self.medical_enhancement = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
                nn.Conv2d(3, 3, kernel_size=1, bias=False),
                nn.Sigmoid(),
            )

    def forward(
        self, images: torch.Tensor, masks: torch.Tensor, return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """医学照片前向传播"""
        # 医学图像预处理
        if self.medical_preprocessing and hasattr(self, "medical_enhancement"):
            batch_size, max_images, channels, height, width = images.shape
            images_flat = images.view(batch_size * max_images, channels, height, width)

            # 应用医学增强
            enhanced_images = self.medical_enhancement(images_flat)
            images = enhanced_images.view(
                batch_size, max_images, channels, height, width
            )

        # 调用父类方法
        return super().forward(images, masks, return_attention)


def create_photo_encoder(config: Dict) -> PhotoEncoder:
    """创建照片编码器工厂函数"""
    encoder_type = config.get("encoder_type", "standard")

    if encoder_type == "medical":
        return MedicalPhotoEncoder(config)
    else:
        return PhotoEncoder(config)


if __name__ == "__main__":
    # 测试照片编码器
    config = {
        "backbone": "vit_base_patch16_224",
        "pretrained": True,
        "feature_dim": 768,
        "num_heads": 8,
        "dropout": 0.1,
        "max_images": 5,
    }

    # 创建编码器
    encoder = create_photo_encoder(config)

    # 测试数据
    batch_size = 2
    max_images = 5
    test_images = torch.randn(batch_size, max_images, 3, 224, 224)
    test_masks = torch.ones(batch_size, max_images, dtype=torch.bool)
    test_masks[0, 3:] = False  # 第一个样本只有3张图片
    test_masks[1, 4:] = False  # 第二个样本只有4张图片

    # 前向传播测试
    result = encoder(test_images, test_masks, return_attention=True)

    print("照片编码结果:")
    print(f"  聚合特征形状: {result['features'].shape}")
    print(f"  单张图片特征形状: {result['individual_features'].shape}")
    print(f"  注意力权重形状: {result['attention_weights'].shape}")
    print(f"  注意力权重示例: {result['attention_weights'][0].cpu().numpy()}")

    # 测试单张图片编码
    single_image = torch.randn(3, 224, 224)
    single_features = encoder.encode_single_image(single_image)
    print(f"  单张图片特征形状: {single_features.shape}")

    print("照片编码器测试完成!")
