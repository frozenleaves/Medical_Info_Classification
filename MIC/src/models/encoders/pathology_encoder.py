"""
病理切片编码器模块
使用多实例学习(MIL)方法处理WSI病理图像
支持Patch-based特征提取和注意力池化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np


class AttentionPooling(nn.Module):
    """
    注意力池化模块
    用于从多个patch特征中提取全局特征
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: Optional[int] = None,
                 num_heads: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim // 4
        self.num_heads = num_heads
        
        # 注意力网络
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_heads),
            nn.Softmax(dim=1)
        )
        
        # 门控机制（可选）
        self.use_gating = True
        if self.use_gating:
            self.gating_net = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, input_dim),
                nn.Sigmoid()
            )
    
    def forward(self, 
                features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: [batch_size, num_patches, feature_dim]
            mask: [batch_size, num_patches] patch有效性掩码
            
        Returns:
            pooled_features: [batch_size, feature_dim]
            attention_weights: [batch_size, num_patches, num_heads]
        """
        batch_size, num_patches, feature_dim = features.shape
        
        # 计算注意力权重
        attention_weights = self.attention_net(features)  # [batch_size, num_patches, num_heads]
        
        # 应用掩码
        if mask is not None:
            # 扩展mask以匹配attention heads
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, self.num_heads)
            attention_weights = attention_weights.masked_fill(~expanded_mask, 0)
            
            # 重新归一化
            attention_sum = attention_weights.sum(dim=1, keepdim=True)
            attention_weights = attention_weights / (attention_sum + 1e-8)
        
        # 门控机制
        if self.use_gating:
            gates = self.gating_net(features)  # [batch_size, num_patches, feature_dim]
            gated_features = features * gates
        else:
            gated_features = features
        
        # 多头注意力池化
        pooled_features = []
        for head in range(self.num_heads):
            head_weights = attention_weights[:, :, head].unsqueeze(-1)  # [batch_size, num_patches, 1]
            head_pooled = torch.sum(gated_features * head_weights, dim=1)  # [batch_size, feature_dim]
            pooled_features.append(head_pooled)
        
        # 合并多头结果
        if self.num_heads > 1:
            pooled_features = torch.stack(pooled_features, dim=1).mean(dim=1)
        else:
            pooled_features = pooled_features[0]
        
        return pooled_features, attention_weights


class MILClassifier(nn.Module):
    """
    多实例学习分类器
    实现经典的MIL方法
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 256,
                 num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 实例级分类器
        self.instance_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 包级聚合
        self.attention_pooling = AttentionPooling(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2
        )
    
    def forward(self, 
                features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: [batch_size, num_patches, feature_dim]
            mask: [batch_size, num_patches]
            
        Returns:
            包含分类结果的字典
        """
        # 实例级预测
        instance_logits = self.instance_classifier(features)  # [batch_size, num_patches, num_classes]
        
        # 注意力池化
        bag_features, attention_weights = self.attention_pooling(features, mask)
        
        # 包级预测
        bag_logits = self.instance_classifier(bag_features.unsqueeze(1)).squeeze(1)
        
        return {
            'bag_logits': bag_logits,
            'instance_logits': instance_logits,
            'attention_weights': attention_weights,
            'bag_features': bag_features
        }


class PathologyEncoder(nn.Module):
    """
    病理切片编码器
    使用patch-based MIL方法处理WSI
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 基本配置
        self.patch_backbone = config.get('patch_backbone', 'vit_small_patch16_224')
        self.pretrained = config.get('pretrained', True)
        self.patch_feature_dim = config.get('patch_feature_dim', 384)
        self.mil_feature_dim = config.get('mil_feature_dim', 512)
        self.attention_heads = config.get('attention_heads', 4)
        self.dropout = config.get('dropout', 0.1)
        self.max_patches = config.get('max_patches', 1000)
        
        # Patch特征提取器
        self.patch_extractor = self._load_patch_extractor()
        
        # 特征投影层
        extractor_dim = self._get_extractor_dim()
        if extractor_dim != self.patch_feature_dim:
            self.patch_projection = nn.Linear(extractor_dim, self.patch_feature_dim)
        else:
            self.patch_projection = nn.Identity()
        
        # MIL聚合模块
        self.mil_aggregator = self._create_mil_aggregator()
        
        # 最终特征投影
        self.feature_projection = nn.Linear(self.patch_feature_dim, self.mil_feature_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.mil_feature_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        print(f"病理切片编码器初始化完成:")
        print(f"  Patch提取器: {self.patch_backbone}")
        print(f"  Patch特征维度: {self.patch_feature_dim}")
        print(f"  MIL特征维度: {self.mil_feature_dim}")
        print(f"  注意力头数: {self.attention_heads}")
        print(f"  最大Patch数: {self.max_patches}")
    
    def _load_patch_extractor(self) -> nn.Module:
        """加载patch特征提取器"""
        from pathlib import Path
        
        # 检查是否是本地路径
        if isinstance(self.patch_backbone, str) and (
            self.patch_backbone.startswith('/') or 
            self.patch_backbone.startswith('./') or 
            Path(self.patch_backbone).exists()
        ):
            # 本地路径处理
            return self._load_local_patch_extractor()
        else:
            # 使用timm加载预训练模型
            return self._load_timm_patch_extractor()
    
    def _load_local_patch_extractor(self) -> nn.Module:
        """加载本地patch提取器模型"""
        from pathlib import Path
        import warnings
        
        model_path = Path(self.patch_backbone)
        
        try:
            print(f"尝试从本地路径加载patch提取器: {model_path}")
            
            # 方法1: 尝试使用transformers库加载
            try:
                from transformers import AutoModel, AutoImageProcessor
                model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
                
                # 添加timm模型兼容属性
                if hasattr(model.config, 'hidden_size'):
                    model.num_features = model.config.hidden_size
                elif hasattr(model, 'embed_dim'):
                    model.num_features = model.embed_dim
                else:
                    model.num_features = 768  # 默认值
                
                # 添加forward方法以兼容timm接口（用于patch处理）
                original_forward = model.forward
                def timm_compatible_forward(x):
                    # 如果是transformers模型，需要适配输入格式
                    if hasattr(model, 'embeddings'):
                        # 对于ViT类模型
                        outputs = original_forward(pixel_values=x)
                        if hasattr(outputs, 'last_hidden_state'):
                            # 取平均池化作为全局特征
                            return outputs.last_hidden_state.mean(dim=1)
                        elif hasattr(outputs, 'pooler_output'):
                            return outputs.pooler_output
                        else:
                            return outputs[0] if isinstance(outputs, tuple) else outputs
                    else:
                        return original_forward(x)
                
                model.forward = timm_compatible_forward
                
                print(f"成功使用transformers加载本地patch提取器, 特征维度: {model.num_features}")
                return model
                
            except ImportError:
                print("transformers库未安装，尝试其他方法...")
            except Exception as e:
                print(f"transformers加载失败: {e}")
            
            # 方法2: 根据路径名推断模型类型
            path_name = model_path.name.lower()
            if 'vit' in path_name:
                # 尝试匹配ViT模型
                if 'base' in path_name:
                    model_name = 'vit_base_patch16_224'
                elif 'large' in path_name:
                    model_name = 'vit_large_patch16_224'
                elif 'small' in path_name:
                    model_name = 'vit_small_patch16_224'
                else:
                    model_name = 'vit_base_patch16_224'
                
                print(f"根据路径名推断patch提取器类型，使用 {model_name}")
                model = timm.create_model(
                    model_name,
                    pretrained=self.pretrained,
                    num_classes=0,
                    global_pool='avg'
                )
                return model
            elif 'resnet' in path_name:
                # 推断ResNet模型
                if '18' in path_name:
                    model_name = 'resnet18'
                elif '50' in path_name:
                    model_name = 'resnet50'
                elif '101' in path_name:
                    model_name = 'resnet101'
                else:
                    model_name = 'resnet18'
                
                print(f"根据路径名推断patch提取器类型，使用 {model_name}")
                model = timm.create_model(
                    model_name,
                    pretrained=self.pretrained,
                    num_classes=0,
                    global_pool='avg'
                )
                return model
                
        except Exception as e:
            print(f"本地patch提取器加载失败: {e}")
        
        # 最后的回退方案
        print("回退到默认的resnet18模型")
        return self._load_timm_patch_extractor('resnet18')
    
    def _load_timm_patch_extractor(self, model_name: str = None) -> nn.Module:
        """使用timm加载patch提取器"""
        model_name = model_name or self.patch_backbone
        
        try:
            model = timm.create_model(
                model_name,
                pretrained=self.pretrained,
                num_classes=0,  # 移除分类头
                global_pool='avg'  # 使用全局平均池化
            )
            
            print(f"成功加载patch提取器: {model_name}")
            return model
            
        except Exception as e:
            print(f"警告: 无法加载 {model_name}: {e}")
            print("回退到 resnet18")
            
            # 回退方案
            model = timm.create_model(
                'resnet18',
                pretrained=True,
                num_classes=0,
                global_pool='avg'
            )
            return model
    
    def _get_extractor_dim(self) -> int:
        """获取特征提取器的输出维度"""
        return self.patch_extractor.num_features
    
    def _create_mil_aggregator(self) -> nn.Module:
        """创建MIL聚合模块"""
        aggregator_type = self.config.get('mil_aggregator', 'attention')
        
        if aggregator_type == 'attention':
            return AttentionPooling(
                input_dim=self.patch_feature_dim,
                hidden_dim=self.patch_feature_dim // 4,
                num_heads=self.attention_heads,
                dropout=self.dropout
            )
        elif aggregator_type == 'max':
            return MaxPooling()
        elif aggregator_type == 'mean':
            return MeanPooling()
        else:
            raise ValueError(f"不支持的MIL聚合器类型: {aggregator_type}")
    
    def forward(self, 
                patches: torch.Tensor, 
                masks: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            patches: [batch_size, max_patches, 3, patch_size, patch_size]
            masks: [batch_size, max_patches] patch有效性掩码
            return_attention: 是否返回注意力权重
            
        Returns:
            包含编码特征的字典
        """
        batch_size, max_patches, channels, patch_h, patch_w = patches.shape
        
        # 将patch维度展平用于特征提取
        patches_flat = patches.view(batch_size * max_patches, channels, patch_h, patch_w)
        
        # 提取patch特征
        with torch.amp.autocast('cuda', enabled=False):  # 某些模型对mixed precision敏感
            patch_features = self.patch_extractor(patches_flat)
            
            # 确保输出是2D张量 [batch, features]
            if len(patch_features.shape) > 2:
                # 如果patch_extractor没有应用全局池化，手动处理
                patch_features = F.adaptive_avg_pool2d(patch_features, (1, 1))
                patch_features = patch_features.flatten(1)
        
        # 特征投影
        patch_features = self.patch_projection(patch_features)
        
        # 重塑为原始维度
        patch_features = patch_features.view(batch_size, max_patches, self.patch_feature_dim)
        
        # MIL聚合
        if isinstance(self.mil_aggregator, AttentionPooling):
            bag_features, attention_weights = self.mil_aggregator(patch_features, masks)
        else:
            bag_features = self.mil_aggregator(patch_features, masks)
            attention_weights = None
        
        # 最终特征投影
        final_features = self.feature_projection(bag_features)
        final_features = self.layer_norm(final_features)
        final_features = self.dropout_layer(final_features)
        
        result = {
            'features': final_features,  # [batch_size, mil_feature_dim]
            'patch_features': patch_features,  # [batch_size, max_patches, patch_feature_dim]
            'masks': masks
        }
        
        if return_attention and attention_weights is not None:
            result['attention_weights'] = attention_weights
        
        return result
    
    def extract_patch_features(self, patches: torch.Tensor) -> torch.Tensor:
        """仅提取patch特征（用于预处理）"""
        self.eval()
        with torch.no_grad():
            batch_size, num_patches = patches.shape[:2]
            patches_flat = patches.view(-1, *patches.shape[2:])
            
            # 分批处理以节省内存
            batch_size_extract = 32
            all_features = []
            
            for i in range(0, patches_flat.shape[0], batch_size_extract):
                batch_patches = patches_flat[i:i+batch_size_extract]
                features = self.patch_extractor(batch_patches)
                features = self.patch_projection(features)
                all_features.append(features)
            
            all_features = torch.cat(all_features, dim=0)
            return all_features.view(batch_size, num_patches, -1)
    
    def get_attention_heatmap(self, 
                             patches: torch.Tensor, 
                             masks: torch.Tensor,
                             coordinates: List[List[Tuple]]) -> List[np.ndarray]:
        """
        生成注意力热力图
        
        Args:
            patches: patch tensor
            masks: patch masks
            coordinates: 每个patch在原图中的坐标
            
        Returns:
            每个样本的注意力热力图
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(patches, masks, return_attention=True)
            attention_weights = result.get('attention_weights')
            
            if attention_weights is None:
                return []
            
            # 处理每个样本
            heatmaps = []
            for i, coords in enumerate(coordinates):
                if len(coords) == 0:
                    continue
                
                sample_attention = attention_weights[i, :len(coords), 0].cpu().numpy()  # 取第一个head
                
                # 创建热力图（这里简化处理）
                # 实际应用中需要根据coordinates重建到原图尺寸
                heatmap = np.zeros((len(coords),))
                for j, weight in enumerate(sample_attention):
                    heatmap[j] = weight
                
                heatmaps.append(heatmap)
            
            return heatmaps


class MaxPooling(nn.Module):
    """最大池化聚合"""
    
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            features = features.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        return features.max(dim=1)[0]


class MeanPooling(nn.Module):
    """平均池化聚合"""
    
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            masked_features = features * mask.unsqueeze(-1).float()
            return masked_features.sum(dim=1) / (mask.sum(dim=1, keepdim=True).float() + 1e-8)
        return features.mean(dim=1)


class TransMIL(nn.Module):
    """
    Transformer-based MIL (简化版)
    参考TransMIL论文实现
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 分类token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 位置编码（简化版）
        self.use_pos_encoding = True
        if self.use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, 1000 + 1, hidden_dim))  # +1 for CLS
    
    def forward(self, 
                features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [batch_size, num_patches, input_dim]
            mask: [batch_size, num_patches]
        Returns:
            bag_features: [batch_size, hidden_dim]
        """
        batch_size, num_patches = features.shape[:2]
        
        # 输入投影
        features = self.input_projection(features)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        features = torch.cat([cls_tokens, features], dim=1)
        
        # 添加位置编码
        if self.use_pos_encoding:
            seq_len = features.shape[1]
            features = features + self.pos_encoding[:, :seq_len, :]
        
        # 创建attention mask
        if mask is not None:
            # 为CLS token添加mask (始终为True)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            
            # Transformer需要的mask格式
            attention_mask = ~full_mask  # True表示需要mask的位置
        else:
            attention_mask = None
        
        # Transformer编码
        encoded_features = self.transformer_encoder(
            features, 
            src_key_padding_mask=attention_mask
        )
        
        # 返回CLS token的特征
        bag_features = encoded_features[:, 0, :]  # [batch_size, hidden_dim]
        
        return bag_features


def create_pathology_encoder(config: Dict) -> PathologyEncoder:
    """创建病理编码器工厂函数"""
    return PathologyEncoder(config)


if __name__ == "__main__":
    # 测试病理编码器
    config = {
        'patch_backbone': 'resnet18',  # 使用轻量级模型测试
        'pretrained': True,
        'patch_feature_dim': 256,
        'mil_feature_dim': 512,
        'attention_heads': 4,
        'dropout': 0.1,
        'max_patches': 100,
        'mil_aggregator': 'attention'
    }
    
    # 创建编码器
    encoder = create_pathology_encoder(config)
    
    # 测试数据
    batch_size = 2
    max_patches = 20
    test_patches = torch.randn(batch_size, max_patches, 3, 224, 224)
    test_masks = torch.ones(batch_size, max_patches, dtype=torch.bool)
    test_masks[0, 15:] = False  # 第一个样本有15个有效patch
    test_masks[1, 18:] = False  # 第二个样本有18个有效patch
    
    # 前向传播测试
    result = encoder(test_patches, test_masks, return_attention=True)
    
    print("病理编码结果:")
    print(f"  聚合特征形状: {result['features'].shape}")
    print(f"  Patch特征形状: {result['patch_features'].shape}")
    print(f"  注意力权重形状: {result['attention_weights'].shape}")
    
    # 测试注意力可视化
    dummy_coords = [[(i*10, j*10) for i in range(15) for j in range(1)],
                    [(i*10, j*10) for i in range(18) for j in range(1)]]
    
    heatmaps = encoder.get_attention_heatmap(test_patches, test_masks, dummy_coords)
    print(f"  生成了 {len(heatmaps)} 个注意力热力图")
    
    print("病理编码器测试完成!")
