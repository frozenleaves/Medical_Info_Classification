# 多模态医学图像分类系统

一个基于深度学习的多模态医学图像分类系统，能够同时处理病历文本、口腔照片和病理切片数据，为医学诊断提供智能化支持。

## 🌟 系统特点

### 📋 多模态数据支持
- **文本模态**: 使用Qwen3-embedding模型处理病历文本
- **图像模态**: 采用ViT+多图片注意力机制处理口腔照片
- **病理模态**: 基于MIL(多实例学习)方法处理WSI病理切片

### 🏗️ 先进架构设计
- **特征提取**: 三种专门的编码器分别处理不同模态数据
- **融合机制**: Transformer-based跨模态注意力融合
- **分类器**: 支持多种分类策略和不确定性估计
- **损失函数**: 多任务学习结合辅助损失和对比学习

### 💡 智能特性
- **注意力机制**: 自动学习不同模态和图片的重要性权重
- **模态自适应**: 支持缺失模态的情况，动态调整权重
- **可解释性**: 提供注意力可视化和特征重要性分析
- **鲁棒性**: 数据增强、模态dropout等提高模型泛化能力

## 📁 项目结构

```
Medical_Image_Classification/
├── MIC/                        # 主要包目录
│   ├── __init__.py            # 包初始化
│   └── src/                   # 源代码目录
│       ├── __init__.py        # 源码包初始化
│       ├── config/            # 配置文件
│       │   ├── __init__.py
│       │   ├── model_config.py         # 模型配置
│       │   └── training_config.py      # 训练配置
│       ├── data/              # 数据处理模块
│       │   ├── __init__.py
│       │   ├── dataset.py     # 数据集类
│       │   ├── dataloader.py  # 数据加载器
│       │   └── transforms.py  # 数据变换
│       ├── models/            # 模型定义
│       │   ├── __init__.py
│       │   ├── encoders/      # 编码器模块
│       │   │   ├── __init__.py
│       │   │   ├── text_encoder.py    # 文本编码器
│       │   │   ├── photo_encoder.py   # 照片编码器
│       │   │   └── pathology_encoder.py # 病理切片编码器
│       │   ├── fusion/        # 多模态融合模块
│       │   │   ├── __init__.py
│       │   │   └── multimodal_fusion.py # 多模态融合
│       │   ├── classifier.py  # 分类器
│       │   └── multimodal_model.py    # 主模型
│       ├── training/          # 训练系统
│       │   ├── __init__.py
│       │   ├── trainer.py     # 训练器
│       │   ├── loss.py        # 损失函数
│       │   └── metrics.py     # 评估指标
│       ├── utils/             # 工具函数
│       │   ├── __init__.py
│       │   ├── data_utils.py  # 数据工具
│       │   └── model_utils.py # 模型工具
│       ├── main.py            # 内部主训练脚本
│       └── inference.py       # 内部推理脚本
├── main.py                    # 主训练脚本入口
├── inference.py               # 推理脚本入口
├── setup.py                   # 包安装脚本
├── MANIFEST.in                # 包文件清单
├── requirements.txt           # 依赖列表
└── README.md                  # 说明文档
```

## 🚀 快速开始

### 环境安装

#### 方式1: 直接安装依赖
```bash
# 克隆项目
git clone https://github.com/your-org/medical-image-classification.git
cd medical-image-classification

# 安装依赖
pip install -r requirements.txt
```

#### 方式2: 安装为包 (推荐)
```bash
# 克隆项目
git clone https://github.com/your-org/medical-image-classification.git
cd medical-image-classification

# 开发模式安装
pip install -e .

# 或者直接安装
pip install .
```

#### 方式3: 从源码安装
```bash
# 直接从GitHub安装
pip install git+https://github.com/your-org/medical-image-classification.git
```

### 数据准备

#### 1. 数据目录结构

```
your_data_dir/
├── texts/                     # 文本文件
│   ├── patient_001.txt
│   ├── patient_002.txt
│   └── ...
├── photos/                    # 口腔照片
│   ├── patient_001/
│   │   ├── photo1.png
│   │   ├── photo2.png
│   │   └── ...
│   ├── patient_002/
│   └── ...
├── pathology/                 # 病理切片
│   ├── patient_001/
│   │   ├── slide1.tiff
│   │   └── ...
│   ├── patient_002/
│   └── ...
└── labels.json               # 标签文件
```

#### 2. 创建数据索引

```bash
# 使用工具函数创建数据索引
python -c "
from utils.data_utils import prepare_data_splits
prepare_data_splits('your_data_dir', label_file='your_data_dir/labels.json')
"
```

#### 3. 标签文件格式

```json
{
  "patient_001": "class_A",
  "patient_002": "class_B",
  ...
}
```

### 模型训练

#### 方式1: 使用入口脚本

```bash
# 基础训练
python main.py --data_dir your_data_dir --experiment_name exp1

# 自定义配置训练
python main.py \
  --data_dir your_data_dir \
  --experiment_name custom_exp \
  --epochs 50 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --num_classes 6 \
  --use_wandb \
  --use_amp

# 恢复训练
python main.py \
  --mode resume \
  --data_dir your_data_dir \
  --checkpoint checkpoints/best_model.pth
```

#### 方式2: 使用命令行工具 (安装包后)

```bash
# 训练
mic-train --data_dir your_data_dir --experiment_name exp1

# 推理
mic-inference --model_path checkpoints/best_model.pth --text "患者症状"
```

#### 方式3: 作为Python包导入

```python
from MIC import create_multimodal_model, MultiModalTrainer

# 创建模型
config = {...}  # 配置字典
model = create_multimodal_model(config)

# 创建训练器
trainer = MultiModalTrainer(model_config, training_config)
```

### 模型推理

#### 单样本推理

```bash
python inference.py \
  --model_path checkpoints/best_model.pth \
  --text "患者口腔内出现红肿症状" \
  --images photo1.png photo2.png \
  --pathology slide1.tiff \
  --visualize
```

#### 批量推理

```python
from MIC.src.inference import MultiModalInferencer

# 初始化推理器
inferencer = MultiModalInferencer('checkpoints/best_model.pth')

# 准备批量数据
samples = [
    {
        'text': '患者A的病历文本',
        'image_paths': ['pathA/img1.png', 'pathA/img2.png'],
        'pathology_paths': ['pathA/slide1.tiff']
    },
    # 更多样本...
]

# 批量预测
results = inferencer.predict_batch(samples)
```

## 💻 开发指南

### 开发环境设置

```bash
# 克隆项目
git clone https://github.com/your-org/medical-image-classification.git
cd medical-image-classification

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 开发模式安装
pip install -e ".[dev]"

# 安装pre-commit钩子
pre-commit install
```

### 代码风格

项目使用以下工具维护代码质量：

- **Black**: 代码格式化
- **Flake8**: 代码风格检查
- **isort**: 导入排序

```bash
# 格式化代码
black MIC/

# 检查代码风格
flake8 MIC/

# 排序导入
isort MIC/
```

### 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest MIC/tests/test_models.py

# 生成覆盖率报告
pytest --cov=MIC --cov-report=html
```

## ⚙️ 配置说明

### 模型配置

```python
model_config = {
    'num_classes': 6,                    # 分类数目
    'use_auxiliary_loss': True,          # 使用辅助损失
    
    # 文本编码器配置
    'text_encoder': {
        'model_name': 'bert-base-chinese',
        'embedding_dim': 768,
        'max_length': 512
    },
    
    # 照片编码器配置
    'photo_encoder': {
        'backbone': 'vit_base_patch16_224',
        'feature_dim': 768,
        'num_heads': 8,
        'max_images': 10
    },
    
    # 病理编码器配置
    'pathology_encoder': {
        'patch_size': 256,
        'patch_backbone': 'vit_small_patch16_224',
        'mil_feature_dim': 512,
        'max_patches': 1000
    },
    
    # 融合模块配置
    'fusion': {
        'fusion_dim': 512,
        'fusion_strategy': 'transformer',  # 'transformer', 'concatenation', 'bilinear'
        'attention_heads': 8
    }
}
```

### 训练配置

```python
training_config = {
    'epochs': 100,
    'learning_rate': 2e-4,
    'batch_size': 4,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    
    # 损失函数配置
    'loss': {
        'type': 'multimodal',
        'main_loss_type': 'focal',        # 'focal', 'cross_entropy', 'label_smoothing'
        'loss_weights': {
            'main_loss': 1.0,
            'text_aux': 0.1,
            'photo_aux': 0.1,
            'pathology_aux': 0.1
        }
    },
    
    # 早停配置
    'early_stopping': {
        'patience': 15,
        'monitor': 'val_f1'
    }
}
```

## 📊 性能监控

### 训练监控

系统支持多种监控方式：

1. **控制台输出**: 实时显示训练进度和指标
2. **Weights & Biases**: 使用`--use_wandb`启用在线实验跟踪
3. **本地日志**: 自动保存训练日志到`experiments/logs/`

### 评估指标

- **分类指标**: 准确率、精确率、召回率、F1分数
- **医学指标**: 敏感性、特异性、AUC、诊断优势比
- **多模态指标**: 各模态独立性能、模态重要性权重
- **类别平衡**: 平衡准确率、Cohen's Kappa

## 🔧 高级功能

### 1. 模型集成

```python
from models.multimodal_model import create_multimodal_model

# 创建多个模型进行集成
models = []
for i in range(3):
    model = create_multimodal_model(config)
    model.load_state_dict(torch.load(f'model_{i}.pth'))
    models.append(model)

# 集成预测
def ensemble_predict(models, batch):
    predictions = []
    for model in models:
        pred = model(batch)['predictions']
        predictions.append(pred)
    
    # 平均集成
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred
```

### 2. 不确定性估计

```python
# 使用不确定性感知分类器
uncertainty_config = {
    'type': 'uncertainty',
    'uncertainty_type': 'epistemic',  # 或 'aleatoric'
    'input_dim': 512,
    'num_classes': 6
}

classifier = create_classifier(uncertainty_config)
```

### 3. 注意力可视化

```python
from inference import MultiModalInferencer

inferencer = MultiModalInferencer('model.pth')
result = inferencer.predict_single(
    text="...", 
    image_paths=["..."],
    return_attention=True
)

# 可视化注意力权重
inferencer.visualize_prediction(result, save_path='attention_viz.png')
```

### 4. 特征提取

```python
# 提取多模态特征用于下游任务
model.eval()
with torch.no_grad():
    outputs = model(batch, return_features=True)
    
    # 获取不同级别的特征
    text_features = outputs['features']['text']          # 文本特征
    photo_features = outputs['features']['photo']        # 照片特征
    pathology_features = outputs['features']['pathology'] # 病理特征
    fused_features = outputs['features']['fused']        # 融合特征
```

## 🎯 应用场景

### 医学诊断辅助
- **口腔疾病分类**: 结合临床文本、口腔照片和组织病理
- **皮肤病诊断**: 整合病史、皮损图像和活检结果
- **肿瘤分型**: 融合影像学、病理学和临床信息

### 研究应用
- **多模态生物标志物发现**: 跨模态特征关联分析
- **疾病亚型研究**: 基于多维数据的精准分层
- **预后评估**: 综合多源信息的风险预测

## 🔬 技术细节

### 核心算法

1. **文本处理**: Transformer-based预训练模型 + 医学领域适应
2. **图像处理**: Vision Transformer + 多图片注意力聚合
3. **病理分析**: 多实例学习 + 注意力池化机制
4. **模态融合**: 跨模态Transformer + 自适应权重学习

### 优化策略

- **混合精度训练**: 加速训练，减少内存占用
- **梯度累积**: 支持大批次训练
- **学习率调度**: Cosine退火 + 线性Warmup
- **正则化**: Dropout、权重衰减、标签平滑

### 鲁棒性保证

- **数据增强**: 文本同义词替换、图像几何变换
- **模态Dropout**: 训练时随机丢弃模态，提高鲁棒性
- **早停机制**: 防止过拟合
- **模型集成**: 多模型投票提高稳定性

## 🐛 常见问题

### Q1: 显存不足怎么办？
```bash
# 减少批次大小和模型尺寸
python main.py --batch_size 2 --accumulation_steps 8
```

### Q2: 数据量较少如何处理？
- 使用预训练模型
- 启用数据增强
- 采用迁移学习策略
- 考虑少样本学习方法

### Q3: 某些模态数据缺失？
系统自动处理缺失模态，无需特殊配置。

### Q4: 如何调整类别权重？
```python
loss_config = {
    'type': 'weighted_ce',
    'class_weights': [1.0, 2.0, 1.5, ...]  # 根据类别频率调整
}
```

## 📈 性能基准

在标准医学数据集上的性能表现：

| 指标 | 单模态(文本) | 单模态(图像) | 单模态(病理) | 多模态融合 |
|------|-------------|-------------|-------------|-----------|
| 准确率 | xx | xx | xx | **xx** |
| F1分数 | xx | xx | xx | **xx** |
| AUC | xx | xx | xx | **xx** |

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 📞 联系我们

如有问题或建议，请通过以下方式联系：

- 🐛 **Bug报告**: 创建Issue
- 💡 **功能建议**: 创建Feature Request
- 📧 **邮件咨询**: your-email@example.com

## 🙏 致谢

感谢以下开源项目的支持：

- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [scikit-learn](https://scikit-learn.org/)

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！
