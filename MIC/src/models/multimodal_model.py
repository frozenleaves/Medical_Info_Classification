"""
多模态医学图像分类主模型
整合所有编码器、融合模块和分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import warnings

from .encoders.text_encoder import create_text_encoder
from .encoders.photo_encoder import create_photo_encoder
from .encoders.pathology_encoder import create_pathology_encoder
from .fusion.multimodal_fusion import create_fusion_module
from .classifier import create_classifier


class MultiModalMedicalClassifier(nn.Module):
    """
    多模态医学信息分类主模型
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 基本配置
        self.num_classes = config.get('num_classes', 6)
        self.use_auxiliary_loss = config.get('use_auxiliary_loss', True)
        
        # 创建编码器
        self.text_encoder = create_text_encoder(config.get('text_encoder', {}))
        self.photo_encoder = create_photo_encoder(config.get('photo_encoder', {}))
        self.pathology_encoder = create_pathology_encoder(config.get('pathology_encoder', {}))
        
        # 创建融合模块
        fusion_config = config.get('fusion', {})
        # 自动设置输入维度
        fusion_config.update({
            'text_dim': self.text_encoder.embedding_dim,
            'photo_dim': self.photo_encoder.feature_dim,
            'pathology_dim': self.pathology_encoder.mil_feature_dim
        })
        self.fusion_module = create_fusion_module(fusion_config)
        
        # 创建主分类器
        classifier_config = config.get('classifier', {})
        classifier_config.update({
            'input_dim': fusion_config.get('fusion_dim', 512),
            'num_classes': self.num_classes
        })
        self.classifier = create_classifier(classifier_config)
        
        # 辅助分类器（用于辅助损失）
        self.auxiliary_classifiers = nn.ModuleDict()
        if self.use_auxiliary_loss:
            # 各模态独立分类器
            self.auxiliary_classifiers['text'] = create_classifier({
                'type': 'mlp',
                'input_dim': self.text_encoder.embedding_dim,
                'num_classes': self.num_classes,
                'hidden_dims': [256, 128],
                'dropout': 0.3
            })
            
            self.auxiliary_classifiers['photo'] = create_classifier({
                'type': 'mlp',
                'input_dim': self.photo_encoder.feature_dim,
                'num_classes': self.num_classes,
                'hidden_dims': [256, 128],
                'dropout': 0.3
            })
            
            self.auxiliary_classifiers['pathology'] = create_classifier({
                'type': 'mlp',
                'input_dim': self.pathology_encoder.mil_feature_dim,
                'num_classes': self.num_classes,
                'hidden_dims': [256, 128],
                'dropout': 0.3
            })
        
        # 模态dropout（训练时随机丢弃模态）
        self.modal_dropout_prob = config.get('modal_dropout_prob', 0.1)
        
        print("多模态医学图像分类模型初始化完成:")
        print(f"  文本编码器维度: {self.text_encoder.embedding_dim}")
        print(f"  照片编码器维度: {self.photo_encoder.feature_dim}")
        print(f"  病理编码器维度: {self.pathology_encoder.mil_feature_dim}")
        print(f"  融合维度: {fusion_config.get('fusion_dim', 512)}")
        print(f"  分类数目: {self.num_classes}")
        print(f"  使用辅助损失: {self.use_auxiliary_loss}")
    
    def forward(self, 
                batch: Dict[str, Any],
                return_features: bool = False,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch: 批次数据字典
            return_features: 是否返回特征
            return_attention: 是否返回注意力权重
            
        Returns:
            包含预测结果的字典
        """
        device = next(self.parameters()).device
        batch_size = len(batch['text']['texts'])
        
        # 检测模态可用性
        modal_availability = self._detect_modal_availability(batch)
        
        # 应用模态dropout（仅训练时）
        if self.training and self.modal_dropout_prob > 0:
            modal_availability = self._apply_modal_dropout(modal_availability)
        
        # 编码各模态特征
        text_result = self._encode_text(batch['text'])
        photo_result = self._encode_photos(batch['photos'], return_attention)
        pathology_result = self._encode_pathology(batch['pathology'], return_attention)
        
        # 提取特征
        text_features = text_result['features']
        photo_features = photo_result['features']
        pathology_features = pathology_result['features']
        
        # 处理缺失模态
        text_features = self._handle_missing_modal(text_features, modal_availability[:, 0])
        photo_features = self._handle_missing_modal(photo_features, modal_availability[:, 1])
        pathology_features = self._handle_missing_modal(pathology_features, modal_availability[:, 2])
        
        # 多模态融合
        fusion_result = self.fusion_module(
            text_features, photo_features, pathology_features, modal_availability
        )
        
        # 主分类
        fused_features = fusion_result['fused_features']
        
        if isinstance(self.classifier, nn.Module):
            main_logits = self.classifier(fused_features)
            if isinstance(main_logits, dict):
                main_logits = main_logits['logits']
        else:
            main_logits = self.classifier(fused_features)
        
        # 构建结果
        result = {
            'logits': main_logits,
            'predictions': F.softmax(main_logits, dim=1),
            'modal_availability': modal_availability
        }
        
        # 辅助分类（如果启用）
        if self.use_auxiliary_loss:
            aux_results = {}
            
            if modal_availability[:, 0].any():  # 有文本模态
                aux_results['text_logits'] = self.auxiliary_classifiers['text'](text_features)
            
            if modal_availability[:, 1].any():  # 有照片模态
                aux_results['photo_logits'] = self.auxiliary_classifiers['photo'](photo_features)
                
            if modal_availability[:, 2].any():  # 有病理模态
                aux_results['pathology_logits'] = self.auxiliary_classifiers['pathology'](pathology_features)
            
            result['auxiliary'] = aux_results
        
        # 返回特征（如果需要）
        if return_features:
            result['features'] = {
                'text': text_features,
                'photo': photo_features,
                'pathology': pathology_features,
                'fused': fused_features,
                'individual': fusion_result.get('individual_features', {}),
                'modal_weights': fusion_result.get('modal_weights', None)
            }
        
        # 返回注意力权重（如果需要）
        if return_attention:
            attention_weights = {}
            
            if 'attention_weights' in photo_result:
                attention_weights['photo'] = photo_result['attention_weights']
                
            if 'attention_weights' in pathology_result:
                attention_weights['pathology'] = pathology_result['attention_weights']
                
            if attention_weights:
                result['attention_weights'] = attention_weights
        
        return result
    
    def _detect_modal_availability(self, batch: Dict[str, Any]) -> torch.Tensor:
        """检测模态可用性"""
        device = next(self.parameters()).device
        batch_size = len(batch['text']['texts'])
        modal_availability = torch.ones(batch_size, 3, dtype=torch.bool, device=device)
        
        # 检测文本模态
        for i, text in enumerate(batch['text']['texts']):
            if not text or not text.strip():
                modal_availability[i, 0] = False
        
        # 检测照片模态
        photo_counts = batch['photos']['counts']
        modal_availability[:, 1] = photo_counts > 0
        
        # 检测病理模态
        pathology_counts = batch['pathology']['counts']
        modal_availability[:, 2] = pathology_counts > 0
        
        return modal_availability
    
    def _apply_modal_dropout(self, modal_availability: torch.Tensor) -> torch.Tensor:
        """应用模态dropout"""
        if not self.training:
            return modal_availability
        
        batch_size = modal_availability.shape[0]
        dropout_mask = torch.rand(batch_size, 3, device=modal_availability.device)
        dropout_mask = dropout_mask > self.modal_dropout_prob
        
        # 确保每个样本至少有一个模态可用
        for i in range(batch_size):
            if not (modal_availability[i] & dropout_mask[i]).any():
                # 随机选择一个原本可用的模态
                available_modals = modal_availability[i].nonzero(as_tuple=True)[0]
                if len(available_modals) > 0:
                    selected_modal = available_modals[torch.randint(0, len(available_modals), (1,))]
                    dropout_mask[i, selected_modal] = True
        
        return modal_availability & dropout_mask
    
    def _encode_text(self, text_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """编码文本数据"""
        try:
            return self.text_encoder(text_data['texts'])
        except Exception as e:
            warnings.warn(f"文本编码失败: {e}")
            device = next(self.parameters()).device
            batch_size = len(text_data['texts'])
            return {
                'features': torch.zeros(batch_size, self.text_encoder.embedding_dim, device=device)
            }
    
    def _encode_photos(self, photo_data: Dict[str, Any], return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """编码照片数据"""
        try:
            return self.photo_encoder(
                photo_data['images'], 
                photo_data['masks'], 
                return_attention
            )
        except Exception as e:
            warnings.warn(f"照片编码失败: {e}")
            device = next(self.parameters()).device
            batch_size = photo_data['images'].shape[0]
            return {
                'features': torch.zeros(batch_size, self.photo_encoder.feature_dim, device=device)
            }
    
    def _encode_pathology(self, pathology_data: Dict[str, Any], return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """编码病理数据"""
        try:
            return self.pathology_encoder(
                pathology_data['patches'],
                pathology_data['masks'],
                return_attention
            )
        except Exception as e:
            warnings.warn(f"病理编码失败: {e}")
            device = next(self.parameters()).device
            batch_size = pathology_data['patches'].shape[0]
            return {
                'features': torch.zeros(batch_size, self.pathology_encoder.mil_feature_dim, device=device)
            }
    
    def _handle_missing_modal(self, features: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        """处理缺失模态"""
        # 将不可用模态的特征置零
        masked_features = features * availability.unsqueeze(-1).float()
        return masked_features
    
    def predict(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """预测接口（推理时使用）"""
        self.eval()
        with torch.no_grad():
            result = self.forward(batch)
            
            # 获取预测类别
            predictions = result['predictions']
            predicted_classes = predictions.argmax(dim=1)
            confidence_scores = predictions.max(dim=1)[0]
            
            return {
                'predicted_classes': predicted_classes.cpu().numpy(),
                'confidence_scores': confidence_scores.cpu().numpy(),
                'class_probabilities': predictions.cpu().numpy(),
                'modal_availability': result['modal_availability'].cpu().numpy()
            }
    
    def get_interpretability_info(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """获取模型可解释性信息"""
        self.eval()
        with torch.no_grad():
            result = self.forward(batch, return_features=True, return_attention=True)
            
            info = {
                'modal_weights': result['features'].get('modal_weights', None),
                'attention_weights': result.get('attention_weights', {}),
                'feature_importance': self._compute_feature_importance(result['features'])
            }
            
            return info
    
    def _compute_feature_importance(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算特征重要性"""
        importance = {}
        
        # 计算各模态特征的L2范数作为重要性指标
        for modal_name, modal_features in features.items():
            if isinstance(modal_features, torch.Tensor):
                importance[modal_name] = torch.norm(modal_features, p=2, dim=1)
        
        return importance


def create_multimodal_model(config: Dict) -> MultiModalMedicalClassifier:
    """创建多模态模型工厂函数"""
    return MultiModalMedicalClassifier(config)


if __name__ == "__main__":
    # 测试多模态模型
    from config.model_config import ModelConfig
    
    # 创建测试配置
    config = {
        'num_classes': 6,
        'use_auxiliary_loss': True,
        'modal_dropout_prob': 0.1,
        
        'text_encoder': {
            'model_name': 'bert-base-chinese',
            'embedding_dim': 768,
            'max_length': 512
        },
        
        'photo_encoder': {
            'backbone': 'vit_base_patch16_224',
            'feature_dim': 768,
            'num_heads': 8,
            'max_images': 5
        },
        
        'pathology_encoder': {
            'patch_backbone': 'resnet18',
            'patch_feature_dim': 256,
            'mil_feature_dim': 512,
            'max_patches': 100
        },
        
        'fusion': {
            'fusion_dim': 512,
            'attention_heads': 8,
            'num_layers': 2,
            'fusion_strategy': 'transformer'
        },
        
        'classifier': {
            'type': 'mlp',
            'hidden_dims': [256, 128],
            'dropout': 0.3
        }
    }
    
    # 创建模型
    model = create_multimodal_model(config)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建测试数据
    batch_size = 2
    test_batch = {
        'text': {
            'texts': ['患者口腔内出现红肿', '建议进行进一步检查'],
            'lengths': torch.tensor([8, 10])
        },
        'photos': {
            'images': torch.randn(batch_size, 5, 3, 224, 224),
            'counts': torch.tensor([3, 4]),
            'masks': torch.ones(batch_size, 5, dtype=torch.bool)
        },
        'pathology': {
            'patches': torch.randn(batch_size, 20, 3, 224, 224),
            'counts': torch.tensor([15, 18]),
            'masks': torch.ones(batch_size, 20, dtype=torch.bool),
            'coordinates': [[(i, j) for i in range(15) for j in range(1)],
                          [(i, j) for i in range(18) for j in range(1)]]
        }
    }
    
    # 前向传播测试
    result = model(test_batch, return_features=True, return_attention=True)
    
    print("\n模型测试结果:")
    print(f"  主要预测形状: {result['logits'].shape}")
    print(f"  预测概率形状: {result['predictions'].shape}")
    print(f"  模态可用性: {result['modal_availability']}")
    
    if 'auxiliary' in result:
        print("  辅助预测:")
        for aux_name, aux_logits in result['auxiliary'].items():
            print(f"    {aux_name}: {aux_logits.shape}")
    
    # 测试预测接口
    predictions = model.predict(test_batch)
    print(f"\n预测结果:")
    print(f"  预测类别: {predictions['predicted_classes']}")
    print(f"  置信度: {predictions['confidence_scores']}")
    
    print("\n多模态模型测试完成!")
