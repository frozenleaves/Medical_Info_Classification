"""
训练配置文件
"""

class TrainingConfig:
    # 数据配置
    data = {
        'batch_size': 4,  # 小batch size适合多模态大模型
        'num_workers': 4,
        'pin_memory': True,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'random_seed': 42
    }
    
    # 训练超参数
    training = {
        'epochs': 100,
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'warmup_epochs': 5,
        'scheduler': 'cosine',
        'gradient_clip': 1.0,
        'accumulation_steps': 4  # 梯度累积
    }
    
    # 损失函数权重
    loss_weights = {
        'main_loss': 1.0,
        'text_aux': 0.1,
        'photo_aux': 0.1,
        'pathology_aux': 0.1
    }
    
    # 早停配置
    early_stopping = {
        'patience': 15,
        'min_delta': 0.001,
        'monitor': 'val_f1'
    }
    
    # 模型保存
    model_save = {
        'save_dir': './checkpoints',
        'save_best': True,
        'save_last': True,
        'save_every_n_epochs': 10
    }
    
    # 数据增强
    augmentation = {
        'text': {
            'enable': True,
            'techniques': ['synonym_replace', 'back_translate']
        },
        'image': {
            'enable': True,
            'techniques': ['rotate', 'flip', 'color_jitter', 'gaussian_blur'],
            'prob': 0.5
        },
        'multimodal': {
            'modal_dropout': 0.1  # 模态dropout概率
        }
    }
