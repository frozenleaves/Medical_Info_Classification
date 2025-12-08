# 快速开始指南

## 🚀 5分钟快速上手

### 1. 环境准备
```bash
# 克隆或下载项目
cd Medical_Image_Classification

# 安装依赖
pip install -r requirements.txt

# 开发模式安装包 (推荐)
pip install -e .
```

### 2. 数据准备
准备数据，目录结构如下：
```
your_data/
├── texts/
│   ├── patient_001.txt
│   └── ...
├── photos/
│   ├── patient_001/
│   │   ├── photo1.png
│   │   └── ...
│   └── ...
├── pathology/
│   ├── patient_001/
│   │   ├── slide1.tiff
│   │   └── ...
│   └── ...
└── labels.json  # {"patient_001": "class_A", ...}
```

创建数据索引：
```bash
python -c "
from MIC.src.utils.data_utils import prepare_data_splits
prepare_data_splits('your_data', label_file='your_data/labels.json')
"
```

### 3. 开始训练
```bash
# 基础训练
python main.py --data_dir your_data --experiment_name my_experiment

# 或使用命令行工具
mic-train --data_dir your_data --experiment_name my_experiment
```

### 4. 模型推理
```bash
# 单样本推理
python inference.py \
  --model_path experiments/my_experiment/checkpoints/best_model.pth \
  --text "患者口腔症状描述" \
  --images photo1.png photo2.png \
  --pathology slide1.tiff

# 或使用命令行工具
mic-inference --model_path best_model.pth --text "症状描述"
```

## 📦 作为Python包使用

```python
# 导入主要组件
from MIC import create_multimodal_model, MultiModalTrainer

# 创建模型
config = {...} 
model = create_multimodal_model(config)

# 训练
trainer = MultiModalTrainer(model_config, training_config)
# ... 训练代码
```

## 🛠️ 常见问题

**Q: 显存不足？**
```bash
# 减少批次大小
python main.py --batch_size 2 --accumulation_steps 8
```

**Q: 缺少某些模态数据？**
- 系统自动处理缺失模态，无需特殊配置

**Q: 想要调整模型结构？**
- 编辑 `MIC/src/config/model_config.py`
- 或在命令行中使用参数覆盖

**Q: 如何可视化结果？**
```bash
# 推理时添加 --visualize 参数
python inference.py --model_path model.pth --visualize
```

## 📚 更多信息
- 详细文档：查看 `README.md`
- 配置选项：查看 `MIC/src/config/`
- 示例数据：查看 `A_Datasets/`（如果存在）

---
✨ 祝您使用愉快！如有问题请创建Issue。
