"""
多模态融合模块
实现跨模态注意力机制和特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制
    计算不同模态之间的相互作用
    """
    
    def __init__(self, 
                 feature_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert self.head_dim * num_heads == feature_dim
        
        # Query, Key, Value投影
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # 输出投影
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, 
                query_features: torch.Tensor,
                key_features: torch.Tensor,
                value_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query_features: [batch_size, feature_dim] 查询模态特征
            key_features: [batch_size, feature_dim] 键模态特征  
            value_features: [batch_size, feature_dim] 值模态特征
            mask: 可选的注意力掩码
            
        Returns:
            attended_features: [batch_size, feature_dim]
            attention_weights: [batch_size, num_heads, 1, 1]
        """
        batch_size = query_features.shape[0]
        
        # 投影到QKV
        Q = self.query_proj(query_features).unsqueeze(1)  # [batch_size, 1, feature_dim]
        K = self.key_proj(key_features).unsqueeze(1)
        V = self.value_proj(value_features).unsqueeze(1)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码（如果有）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, V)
        
        # 合并多头
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, 1, self.feature_dim
        ).squeeze(1)
        
        # 输出投影
        output = self.output_proj(attended)
        
        return output, attention_weights.squeeze()


class MultiModalTransformer(nn.Module):
    """
    多模态Transformer融合模块
    支持自注意力和跨模态注意力
    """
    
    def __init__(self,
                 feature_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 feedforward_dim: Optional[int] = None):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        feedforward_dim = feedforward_dim or feature_dim * 4
        
        # 模态投影层（确保维度一致）
        self.text_projection = nn.Linear(feature_dim, feature_dim)
        self.photo_projection = nn.Linear(feature_dim, feature_dim)
        self.pathology_projection = nn.Linear(feature_dim, feature_dim)
        
        # 位置编码（用于区分不同模态）
        self.modality_embeddings = nn.Parameter(torch.randn(3, feature_dim))  # 3个模态
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出投影
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, 
                text_features: torch.Tensor,
                photo_features: torch.Tensor,
                pathology_features: torch.Tensor,
                modal_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            text_features: [batch_size, feature_dim]
            photo_features: [batch_size, feature_dim]
            pathology_features: [batch_size, feature_dim]
            modal_mask: [batch_size, 3] 模态有效性掩码
            
        Returns:
            融合后的特征字典
        """
        batch_size = text_features.shape[0]
        
        # 模态投影
        text_proj = self.text_projection(text_features)
        photo_proj = self.photo_projection(photo_features)
        pathology_proj = self.pathology_projection(pathology_features)
        
        # 添加模态位置编码
        text_proj = text_proj + self.modality_embeddings[0]
        photo_proj = photo_proj + self.modality_embeddings[1]
        pathology_proj = pathology_proj + self.modality_embeddings[2]
        
        # 拼接所有模态特征
        multimodal_features = torch.stack([text_proj, photo_proj, pathology_proj], dim=1)
        # [batch_size, 3, feature_dim]
        
        # 应用模态掩码
        if modal_mask is not None:
            # modal_mask: [batch_size, 3] -> attention_mask for transformer
            attention_mask = ~modal_mask  # True表示需要mask的位置
        else:
            attention_mask = None
        
        # Transformer编码
        fused_features = self.transformer(
            multimodal_features, 
            src_key_padding_mask=attention_mask
        )  # [batch_size, 3, feature_dim]
        
        # 提取各模态融合后的特征
        fused_text = fused_features[:, 0, :]
        fused_photo = fused_features[:, 1, :]
        fused_pathology = fused_features[:, 2, :]
        
        # 全局融合特征（平均池化）
        if modal_mask is not None:
            # 加权平均
            modal_weights = modal_mask.float()
            modal_weights = modal_weights / (modal_weights.sum(dim=1, keepdim=True) + 1e-8)
            global_features = torch.sum(
                fused_features * modal_weights.unsqueeze(-1), dim=1
            )
        else:
            global_features = fused_features.mean(dim=1)
        
        # 输出投影和归一化
        global_features = self.output_projection(global_features)
        global_features = self.layer_norm(global_features)
        
        return {
            'global_features': global_features,
            'fused_text': fused_text,
            'fused_photo': fused_photo,
            'fused_pathology': fused_pathology,
            'attention_weights': None  # 可以添加注意力权重提取
        }


class BilinearFusion(nn.Module):
    """
    双线性融合模块
    计算两个模态之间的双线性相互作用
    """
    
    def __init__(self, 
                 dim1: int, 
                 dim2: int, 
                 output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim
        
        # 双线性层
        self.bilinear = nn.Bilinear(dim1, dim2, output_dim)
        
        # 额外的线性投影
        self.linear1 = nn.Linear(dim1, output_dim)
        self.linear2 = nn.Linear(dim2, output_dim)
        
        # 激活和dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features1: [batch_size, dim1]
            features2: [batch_size, dim2]
            
        Returns:
            fused_features: [batch_size, output_dim]
        """
        # 双线性融合
        bilinear_output = self.bilinear(features1, features2)
        
        # 线性投影
        linear1_output = self.linear1(features1)
        linear2_output = self.linear2(features2)
        
        # 融合所有输出
        fused = bilinear_output + linear1_output + linear2_output
        fused = self.activation(fused)
        fused = self.dropout(fused)
        
        return fused


class MultiModalFusionModule(nn.Module):
    """
    多模态融合主模块
    整合多种融合策略
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 基本配置
        self.fusion_dim = config.get('fusion_dim', 512)
        self.attention_heads = config.get('attention_heads', 8)
        self.dropout = config.get('dropout', 0.2)
        self.num_layers = config.get('num_layers', 2)
        
        # 输入特征维度
        self.text_dim = config.get('text_dim', 768)
        self.photo_dim = config.get('photo_dim', 768)
        self.pathology_dim = config.get('pathology_dim', 512)
        
        # 特征对齐（投影到统一维度）
        self.text_align = nn.Linear(self.text_dim, self.fusion_dim)
        self.photo_align = nn.Linear(self.photo_dim, self.fusion_dim)
        self.pathology_align = nn.Linear(self.pathology_dim, self.fusion_dim)
        
        # 融合策略选择
        self.fusion_strategy = config.get('fusion_strategy', 'transformer')
        
        if self.fusion_strategy == 'transformer':
            self.fusion_module = MultiModalTransformer(
                feature_dim=self.fusion_dim,
                num_heads=self.attention_heads,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
        elif self.fusion_strategy == 'concatenation':
            self.fusion_module = self._create_concat_fusion()
        elif self.fusion_strategy == 'bilinear':
            self.fusion_module = self._create_bilinear_fusion()
        else:
            raise ValueError(f"不支持的融合策略: {self.fusion_strategy}")
        
        # 自适应权重模块
        self.adaptive_weighting = AdaptiveModalWeighting(self.fusion_dim, 3)
        
        # 最终投影层
        self.final_projection = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_dim, self.fusion_dim)
        )
        
        print(f"多模态融合模块初始化完成:")
        print(f"  融合维度: {self.fusion_dim}")
        print(f"  融合策略: {self.fusion_strategy}")
        print(f"  注意力头数: {self.attention_heads}")
    
    def _create_concat_fusion(self) -> nn.Module:
        """创建拼接融合模块"""
        return nn.Sequential(
            nn.Linear(self.fusion_dim * 3, self.fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
    
    def _create_bilinear_fusion(self) -> nn.Module:
        """创建双线性融合模块"""
        class BilinearFusionWrapper(nn.Module):
            def __init__(self, fusion_dim):
                super().__init__()
                # 两两双线性融合
                self.text_photo_fusion = BilinearFusion(fusion_dim, fusion_dim, fusion_dim)
                self.text_pathology_fusion = BilinearFusion(fusion_dim, fusion_dim, fusion_dim)
                self.photo_pathology_fusion = BilinearFusion(fusion_dim, fusion_dim, fusion_dim)
                
                # 最终融合
                self.final_fusion = nn.Linear(fusion_dim * 3, fusion_dim)
                
            def forward(self, text_features, photo_features, pathology_features, modal_mask=None):
                # 两两融合
                text_photo = self.text_photo_fusion(text_features, photo_features)
                text_pathology = self.text_pathology_fusion(text_features, pathology_features)
                photo_pathology = self.photo_pathology_fusion(photo_features, pathology_features)
                
                # 拼接融合
                combined = torch.cat([text_photo, text_pathology, photo_pathology], dim=1)
                global_features = self.final_fusion(combined)
                
                return {
                    'global_features': global_features,
                    'fused_text': text_features,
                    'fused_photo': photo_features,
                    'fused_pathology': pathology_features
                }
        
        return BilinearFusionWrapper(self.fusion_dim)
    
    def forward(self, 
                text_features: torch.Tensor,
                photo_features: torch.Tensor,
                pathology_features: torch.Tensor,
                modal_availability: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            text_features: [batch_size, text_dim]
            photo_features: [batch_size, photo_dim]
            pathology_features: [batch_size, pathology_dim]
            modal_availability: [batch_size, 3] 模态可用性掩码
            
        Returns:
            融合特征字典
        """
        # 特征对齐
        aligned_text = self.text_align(text_features)
        aligned_photo = self.photo_align(photo_features)
        aligned_pathology = self.pathology_align(pathology_features)
        
        # 特征标准化
        aligned_text = F.normalize(aligned_text, p=2, dim=1)
        aligned_photo = F.normalize(aligned_photo, p=2, dim=1)
        aligned_pathology = F.normalize(aligned_pathology, p=2, dim=1)
        
        # 模态融合
        if self.fusion_strategy == 'transformer':
            fusion_result = self.fusion_module(
                aligned_text, aligned_photo, aligned_pathology, modal_availability
            )
        elif self.fusion_strategy == 'concatenation':
            # 拼接融合
            concatenated = torch.cat([aligned_text, aligned_photo, aligned_pathology], dim=1)
            global_features = self.fusion_module(concatenated)
            
            fusion_result = {
                'global_features': global_features,
                'fused_text': aligned_text,
                'fused_photo': aligned_photo,
                'fused_pathology': aligned_pathology
            }
        elif self.fusion_strategy == 'bilinear':
            fusion_result = self.fusion_module(
                aligned_text, aligned_photo, aligned_pathology, modal_availability
            )
        
        # 自适应加权
        modal_features = torch.stack([
            fusion_result['fused_text'],
            fusion_result['fused_photo'], 
            fusion_result['fused_pathology']
        ], dim=1)  # [batch_size, 3, fusion_dim]
        
        weighted_features = self.adaptive_weighting(modal_features, modal_availability)
        
        # 最终投影
        final_features = self.final_projection(weighted_features)
        
        return {
            'fused_features': final_features,
            'global_features': fusion_result['global_features'],
            'individual_features': {
                'text': fusion_result['fused_text'],
                'photo': fusion_result['fused_photo'],
                'pathology': fusion_result['fused_pathology']
            },
            'modal_weights': self.adaptive_weighting.get_last_weights()
        }


class AdaptiveModalWeighting(nn.Module):
    """
    自适应模态权重模块
    学习不同样本对不同模态的依赖程度
    """
    
    def __init__(self, feature_dim: int, num_modalities: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # 权重预测网络
        self.weight_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        self.last_weights = None
        
    def forward(self, 
                modal_features: torch.Tensor,
                modal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            modal_features: [batch_size, num_modalities, feature_dim]
            modal_mask: [batch_size, num_modalities] 模态可用性
            
        Returns:
            weighted_features: [batch_size, feature_dim]
        """
        # 计算全局特征用于权重预测
        if modal_mask is not None:
            masked_features = modal_features * modal_mask.unsqueeze(-1)
            global_features = masked_features.sum(dim=1) / (modal_mask.sum(dim=1, keepdim=True) + 1e-8)
        else:
            global_features = modal_features.mean(dim=1)
        
        # 预测权重
        modal_weights = self.weight_predictor(global_features)  # [batch_size, num_modalities]
        
        # 应用掩码
        if modal_mask is not None:
            modal_weights = modal_weights * modal_mask
            modal_weights = modal_weights / (modal_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # 加权聚合
        weighted_features = torch.sum(
            modal_features * modal_weights.unsqueeze(-1), dim=1
        )
        
        self.last_weights = modal_weights
        return weighted_features
    
    def get_last_weights(self) -> Optional[torch.Tensor]:
        """获取最后一次的权重"""
        return self.last_weights


def create_fusion_module(config: Dict) -> MultiModalFusionModule:
    """创建融合模块工厂函数"""
    return MultiModalFusionModule(config)


if __name__ == "__main__":
    # 测试融合模块
    config = {
        'fusion_dim': 512,
        'attention_heads': 8,
        'dropout': 0.1,
        'num_layers': 2,
        'text_dim': 768,
        'photo_dim': 768,
        'pathology_dim': 512,
        'fusion_strategy': 'transformer'
    }
    
    # 创建融合模块
    fusion_module = create_fusion_module(config)
    
    # 测试数据
    batch_size = 2
    text_features = torch.randn(batch_size, 768)
    photo_features = torch.randn(batch_size, 768)
    pathology_features = torch.randn(batch_size, 512)
    
    # 模态可用性掩码
    modal_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    modal_mask[0, 2] = False  # 第一个样本没有病理数据
    
    # 前向传播测试
    result = fusion_module(text_features, photo_features, pathology_features, modal_mask)
    
    print("融合模块测试结果:")
    print(f"  融合特征形状: {result['fused_features'].shape}")
    print(f"  全局特征形状: {result['global_features'].shape}")
    print(f"  模态权重: {result['modal_weights']}")
    
    # 测试不同融合策略
    for strategy in ['concatenation', 'bilinear']:
        test_config = config.copy()
        test_config['fusion_strategy'] = strategy
        
        test_module = create_fusion_module(test_config)
        test_result = test_module(text_features, photo_features, pathology_features)
        
        print(f"{strategy} 融合结果: {test_result['fused_features'].shape}")
    
    print("融合模块测试完成!")
