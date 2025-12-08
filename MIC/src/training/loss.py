"""
损失函数模块
实现多种适合多模态医学分类任务的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss
    解决类别不平衡问题
    """
    
    def __init__(self, 
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            inputs: [batch_size, num_classes] 预测logits
            targets: [batch_size] 真实标签
            
        Returns:
            loss: 标量损失值
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        
        # 应用alpha权重
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets.clamp(0))
            ce_loss = at * ce_loss
        
        # 应用focal权重
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    减少过拟合，提高模型泛化能力
    """
    
    def __init__(self, 
                 num_classes: int,
                 smoothing: float = 0.1,
                 reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算标签平滑交叉熵损失
        
        Args:
            inputs: [batch_size, num_classes] 预测logits
            targets: [batch_size] 真实标签
            
        Returns:
            loss: 标量损失值
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # 创建平滑标签
        smooth_targets = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # 计算损失
        loss = -torch.sum(smooth_targets * log_probs, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiModalLoss(nn.Module):
    """
    多模态损失函数
    结合主损失和辅助损失
    """
    
    def __init__(self, 
                 loss_config: Dict[str, Any]):
        super().__init__()
        
        self.loss_weights = loss_config.get('loss_weights', {
            'main_loss': 1.0,
            'text_aux': 0.1,
            'photo_aux': 0.1,
            'pathology_aux': 0.1
        })
        
        # 主损失函数
        main_loss_type = loss_config.get('main_loss_type', 'cross_entropy')
        self.main_loss = self._create_loss_function(main_loss_type, loss_config)
        
        # 辅助损失函数
        aux_loss_type = loss_config.get('aux_loss_type', 'cross_entropy')
        self.aux_loss = self._create_loss_function(aux_loss_type, loss_config)
        
        # 对比学习损失（可选）
        self.use_contrastive_loss = loss_config.get('use_contrastive_loss', False)
        if self.use_contrastive_loss:
            self.contrastive_loss = ContrastiveLoss(
                temperature=loss_config.get('contrastive_temperature', 0.07)
            )
            self.contrastive_weight = loss_config.get('contrastive_weight', 0.1)
    
    def _create_loss_function(self, loss_type: str, config: Dict) -> nn.Module:
        """创建损失函数"""
        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_type == 'focal':
            return FocalLoss(
                alpha=config.get('focal_alpha', None),
                gamma=config.get('focal_gamma', 2.0)
            )
        elif loss_type == 'label_smoothing':
            return LabelSmoothingCrossEntropy(
                num_classes=config.get('num_classes', 6),
                smoothing=config.get('smoothing', 0.1)
            )
        else:
            return nn.CrossEntropyLoss()
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor], 
                targets: torch.Tensor,
                modal_availability: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算多模态损失
        
        Args:
            outputs: 模型输出字典
            targets: [batch_size] 真实标签
            modal_availability: [batch_size, 3] 模态可用性掩码
            
        Returns:
            包含各损失项的字典
        """
        losses = {}
        total_loss = 0.0
        
        # 主损失
        main_logits = outputs['logits']
        main_loss = self.main_loss(main_logits, targets)
        losses['main_loss'] = main_loss
        total_loss += self.loss_weights['main_loss'] * main_loss
        
        # 辅助损失
        if 'auxiliary' in outputs:
            aux_outputs = outputs['auxiliary']
            
            # 文本辅助损失
            if 'text_logits' in aux_outputs:
                if modal_availability is None or modal_availability[:, 0].any():
                    text_aux_loss = self.aux_loss(aux_outputs['text_logits'], targets)
                    losses['text_aux_loss'] = text_aux_loss
                    total_loss += self.loss_weights['text_aux'] * text_aux_loss
            
            # 照片辅助损失
            if 'photo_logits' in aux_outputs:
                if modal_availability is None or modal_availability[:, 1].any():
                    photo_aux_loss = self.aux_loss(aux_outputs['photo_logits'], targets)
                    losses['photo_aux_loss'] = photo_aux_loss
                    total_loss += self.loss_weights['photo_aux'] * photo_aux_loss
            
            # 病理辅助损失
            if 'pathology_logits' in aux_outputs:
                if modal_availability is None or modal_availability[:, 2].any():
                    pathology_aux_loss = self.aux_loss(aux_outputs['pathology_logits'], targets)
                    losses['pathology_aux_loss'] = pathology_aux_loss
                    total_loss += self.loss_weights['pathology_aux'] * pathology_aux_loss
        
        # 对比学习损失（可选）
        if self.use_contrastive_loss and 'features' in outputs:
            contrastive_loss = self._compute_contrastive_loss(outputs['features'], targets)
            losses['contrastive_loss'] = contrastive_loss
            total_loss += self.contrastive_weight * contrastive_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _compute_contrastive_loss(self, 
                                 features: Dict[str, torch.Tensor], 
                                 targets: torch.Tensor) -> torch.Tensor:
        """计算对比学习损失"""
        # 使用融合特征进行对比学习
        if 'fused' in features:
            embeddings = features['fused']
            return self.contrastive_loss(embeddings, targets)
        else:
            return torch.tensor(0.0, device=targets.device)


class ContrastiveLoss(nn.Module):
    """
    对比学习损失（InfoNCE）
    增强同类样本的相似性，降低不同类样本的相似性
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失
        
        Args:
            embeddings: [batch_size, embedding_dim] 特征嵌入
            labels: [batch_size] 标签
            
        Returns:
            loss: 标量损失值
        """
        # 特征归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # 创建标签掩码
        batch_size = embeddings.size(0)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(embeddings.device)
        
        # 移除对角线（自己和自己的相似度）
        mask = mask - torch.eye(batch_size, device=embeddings.device)
        
        # 计算正样本和负样本
        positive_mask = mask
        negative_mask = 1 - mask - torch.eye(batch_size, device=embeddings.device)
        
        # InfoNCE损失
        # exp_similarities = torch.exp(similarity_matrix)
        # positive_sums = torch.sum(exp_similarities * positive_mask, dim=1)
        # negative_sums = torch.sum(exp_similarities * negative_mask, dim=1)
        # 
        # loss = -torch.mean(torch.log(positive_sums / (positive_sums + negative_sums + 1e-8)))
        
        # 简化版本：使用交叉熵
        logits = similarity_matrix
        targets = torch.arange(batch_size, device=embeddings.device)
        loss = F.cross_entropy(logits, targets)
        
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss
    适用于分割任务，也可以用于分类任务中的软标签
    """
    
    def __init__(self, smooth: float = 1e-8):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Dice损失
        
        Args:
            inputs: [batch_size, num_classes] 预测概率
            targets: [batch_size, num_classes] one-hot编码的真实标签
            
        Returns:
            loss: 标量损失值
        """
        inputs = F.softmax(inputs, dim=1)
        
        # 计算Dice系数
        intersection = torch.sum(inputs * targets, dim=0)
        union = torch.sum(inputs, dim=0) + torch.sum(targets, dim=0)
        
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coeff.mean()
        
        return dice_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵损失
    根据类别频率自动调整权重
    """
    
    def __init__(self, 
                 class_weights: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算加权交叉熵损失
        
        Args:
            inputs: [batch_size, num_classes] 预测logits
            targets: [batch_size] 真实标签
            
        Returns:
            loss: 标量损失值
        """
        if self.class_weights is not None and self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
            
        return F.cross_entropy(
            inputs, targets, 
            weight=self.class_weights,
            reduction=self.reduction
        )


def create_loss_function(config: Dict[str, Any]) -> nn.Module:
    """
    创建损失函数工厂函数
    
    Args:
        config: 损失函数配置
        
    Returns:
        损失函数实例
    """
    loss_type = config.get('type', 'multimodal')
    
    if loss_type == 'multimodal':
        return MultiModalLoss(config)
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=config.get('alpha', None),
            gamma=config.get('gamma', 2.0)
        )
    elif loss_type == 'label_smoothing':
        return LabelSmoothingCrossEntropy(
            num_classes=config.get('num_classes', 6),
            smoothing=config.get('smoothing', 0.1)
        )
    elif loss_type == 'weighted_ce':
        return WeightedCrossEntropyLoss(
            class_weights=config.get('class_weights', None)
        )
    elif loss_type == 'contrastive':
        return ContrastiveLoss(
            temperature=config.get('temperature', 0.07)
        )
    else:
        return nn.CrossEntropyLoss()


if __name__ == "__main__":
    # 测试损失函数
    batch_size = 4
    num_classes = 6
    
    # 创建测试数据
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 创建模型输出
    model_outputs = {
        'logits': logits,
        'auxiliary': {
            'text_logits': torch.randn(batch_size, num_classes),
            'photo_logits': torch.randn(batch_size, num_classes),
            'pathology_logits': torch.randn(batch_size, num_classes)
        },
        'features': {
            'fused': torch.randn(batch_size, 512)
        }
    }
    
    # 模态可用性
    modal_availability = torch.ones(batch_size, 3, dtype=torch.bool)
    modal_availability[0, 2] = False  # 第一个样本没有病理数据
    
    # 测试多模态损失
    loss_config = {
        'type': 'multimodal',
        'main_loss_type': 'focal',
        'aux_loss_type': 'cross_entropy',
        'num_classes': num_classes,
        'focal_gamma': 2.0,
        'loss_weights': {
            'main_loss': 1.0,
            'text_aux': 0.1,
            'photo_aux': 0.1,
            'pathology_aux': 0.1
        },
        'use_contrastive_loss': True,
        'contrastive_weight': 0.1
    }
    
    loss_fn = create_loss_function(loss_config)
    losses = loss_fn(model_outputs, targets, modal_availability)
    
    print("损失函数测试结果:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    # 测试其他损失函数
    focal_loss = FocalLoss(gamma=2.0)
    focal_result = focal_loss(logits, targets)
    print(f"\nFocal Loss: {focal_result.item():.4f}")
    
    smooth_ce = LabelSmoothingCrossEntropy(num_classes=num_classes, smoothing=0.1)
    smooth_result = smooth_ce(logits, targets)
    print(f"Label Smoothing CE: {smooth_result.item():.4f}")
    
    print("\n损失函数测试完成!")
