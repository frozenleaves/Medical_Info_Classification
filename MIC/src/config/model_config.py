"""
模型配置
"""

class ModelConfig:
    # 基础配置
    num_classes = 6  # 分类数目
    
    # 文本编码器配置
    text_encoder = {
        'model_name': 'Qwen/Qwen3-Embedding-0.6B',  # 或使用embedding模型
        'embedding_dim': 1024,
        'max_length': 1024,
        'freeze_encoder': False
    }
    
    # 照片编码器配置
    photo_encoder = {
        'backbone': 'vit_base_patch16_224',
        'pretrained': True,
        'feature_dim': 768,
        'num_heads': 8,  # 多图片注意力头数
        'dropout': 0.1,
        'max_images': 10  # 最大图片数量
    }
    
    # 病理切片MIL编码器配置
    pathology_encoder = {
        'patch_size': 256,  # patch尺寸
        'overlap': 0.1,     # 重叠比例
        'patch_backbone': 'vit_large_patch16_224',
        'patch_feature_dim': 1024,
        'mil_feature_dim': 512,
        'attention_heads': 4,
        'dropout': 0.1,
        'max_patches': 1000  # 最大patch数量
    }
    
    # 融合模块配置
    fusion = {
        'fusion_dim': 256,
        'attention_heads': 4,
        'dropout': 0.2,
        'num_layers': 2,
        'fusion_strategy': 'transformer'
    },
    
    # 分类器配置
    classifier = {
        'type': 'mlp',
        'hidden_dims': [128, 64],
        'dropout': 0.3,
        'activation': 'relu',
        'use_batch_norm': True  # 数据少时关闭BatchNorm
    }

