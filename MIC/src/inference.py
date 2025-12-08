"""
推理脚本
用于单个样本或批量样本的预测
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from .models.multimodal_model import create_multimodal_model
from .data.transforms import get_transforms
from .data.dataset import MultiModalMedicalDataset
from .training.metrics import MetricCalculator


class MultiModalInferencer:
    """
    多模态医学图像分类推理器
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'auto',
                 config_path: Optional[str] = None):
        """
        初始化推理器
        
        Args:
            model_path: 模型权重路径
            device: 设备 ('auto', 'cpu', 'cuda')
            config_path: 配置文件路径（可选）
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        
        # 设备选择
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 加载模型和配置
        self.model, self.model_config, checkpoint = self._load_model()
        
        # 设置类别名称（优先从checkpoint加载）
        # 优先级：checkpoint中的实际标签 > model_config中的配置 > 默认值
        model_num_classes = self.model_config.get('num_classes', 6)
        
        if 'class_names' in checkpoint and checkpoint['class_names']:
            loaded_class_names = checkpoint['class_names']
            print(f"从checkpoint加载类别名称: {loaded_class_names}")
            
            # 检查长度是否匹配
            if len(loaded_class_names) != model_num_classes:
                print(f"⚠️  警告: checkpoint中的类别数({len(loaded_class_names)}) "
                      f"与模型定义({model_num_classes})不一致")
                
                # 调整class_names长度以匹配模型
                self.class_names = []
                for i in range(model_num_classes):
                    if i < len(loaded_class_names):
                        self.class_names.append(loaded_class_names[i])
                    else:
                        self.class_names.append(f'Unused_Class_{i}')
                
                print(f"调整后的类别名称: {self.class_names}")
            else:
                self.class_names = loaded_class_names
                
        elif 'label_map' in checkpoint and checkpoint['label_map']:
            # 从label_map构建class_names
            label_map = checkpoint['label_map']
            self.label_map = label_map
            self.idx_to_label = {idx: label for label, idx in label_map.items()}
            
            # 构建class_names，长度与模型一致
            self.class_names = []
            for i in range(model_num_classes):
                if i in self.idx_to_label:
                    self.class_names.append(self.idx_to_label[i])
                else:
                    self.class_names.append(f'Unused_Class_{i}')
            
            print(f"从checkpoint的label_map构建类别名称: {self.class_names}")
            
            if len(label_map) != model_num_classes:
                print(f"⚠️  注意: 数据集有{len(label_map)}个类别，模型定义{model_num_classes}个类别")
        else:
            self.class_names = self.model_config.get('class_names', 
                                                     [f'Class_{i}' for i in range(model_num_classes)])
            print(f"⚠️  使用默认类别名称（可能不准确）: {self.class_names}")
        
        self.num_classes = len(self.class_names)
        
        # 验证长度一致性
        if self.num_classes != model_num_classes:
            print(f"⚠️  严重警告: class_names长度({self.num_classes}) "
                  f"与模型类别数({model_num_classes})不一致！")
            print(f"这可能导致推理错误，请检查模型配置。")
        
        # 保存label_map（如果有）
        if 'label_map' in checkpoint:
            self.label_map = checkpoint['label_map']
        if 'idx_to_label' in checkpoint:
            self.idx_to_label = checkpoint['idx_to_label']
        
        # 创建数据变换
        transform_config = self.model_config.get('transforms', {})
        self.transforms = get_transforms(transform_config, is_training=False)
        
        print(f"推理器初始化完成:")
        print(f"  类别数: {self.num_classes}")
        print(f"  类别名称: {self.class_names}")
    
    def _load_model(self) -> tuple:
        """加载模型和配置"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 获取模型配置
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        elif self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                model_config = json.load(f)
        else:
            raise ValueError("无法找到模型配置，请提供config_path或确保模型文件包含配置")
        
        # 创建模型
        model = create_multimodal_model(model_config)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"模型加载完成: {self.model_path}")
        
        return model, model_config, checkpoint
    
    def predict_single(self, 
                      text: str = "",
                      image_paths: List[str] = None,
                      pathology_paths: List[str] = None,
                      return_attention: bool = False,
                      return_features: bool = False) -> Dict[str, Any]:
        """
        单样本预测
        
        Args:
            text: 病历文本
            image_paths: 口腔照片路径列表
            pathology_paths: 病理切片路径列表  
            return_attention: 是否返回注意力权重
            return_features: 是否返回特征
            
        Returns:
            预测结果字典
        """
        # 准备输入数据
        batch = self._prepare_single_sample(text, image_paths, pathology_paths)
        
        # 预测
        with torch.no_grad():
            outputs = self.model.forward(
                batch, 
                return_features=return_features,
                return_attention=return_attention
            )
        
        # 处理预测结果
        predictions = outputs['predictions'][0]  # 取第一个样本
        predicted_class = predictions.argmax().item()
        confidence = predictions[predicted_class].item()
        
        result = {
            'predicted_class': predicted_class,
            'predicted_class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'class_probabilities': {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(predictions)
            },
            'modal_availability': outputs['modal_availability'][0].cpu().numpy().tolist()
        }
        
        # 添加注意力权重（如果请求）
        if return_attention and 'attention_weights' in outputs:
            result['attention_weights'] = self._process_attention_weights(outputs['attention_weights'])
        
        # 添加特征（如果请求）
        if return_features and 'features' in outputs:
            result['features'] = self._process_features(outputs['features'])
        
        return result
    
    def predict_batch(self, 
                     samples: List[Dict[str, Any]],
                     batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            samples: 样本列表，每个样本包含 {'text', 'image_paths', 'pathology_paths'}
            batch_size: 批次大小
            
        Returns:
            预测结果列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            
            # 准备批次数据
            batch = self._prepare_batch_samples(batch_samples)
            
            # 预测
            with torch.no_grad():
                outputs = self.model.forward(batch)
            
            # 处理每个样本的结果
            for j, sample_output in enumerate(self._split_batch_outputs(outputs)):
                predictions = sample_output['predictions']
                predicted_class = predictions.argmax().item()
                confidence = predictions[predicted_class].item()
                
                result = {
                    'sample_index': i + j,
                    'predicted_class': predicted_class,
                    'predicted_class_name': self.class_names[predicted_class],
                    'confidence': confidence,
                    'class_probabilities': {
                        self.class_names[k]: prob.item() 
                        for k, prob in enumerate(predictions)
                    }
                }
                
                results.append(result)
        
        return results
    
    def _prepare_single_sample(self, 
                              text: str, 
                              image_paths: List[str], 
                              pathology_paths: List[str]) -> Dict[str, Any]:
        """准备单个样本的输入数据"""
        # 文本处理
        processed_text = text.strip() if text else ""
        if 'text' in self.transforms:
            processed_text = self.transforms['text'](processed_text)
        
        # 图片处理
        images = []
        if image_paths:
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    if 'photo' in self.transforms:
                        img_tensor = self.transforms['photo'](img)
                        images.append(img_tensor)
                except Exception as e:
                    warnings.warn(f"加载图片失败 {img_path}: {e}")
        
        # 病理切片处理（简化版，实际中可能需要patch提取）
        pathology_patches = []
        if pathology_paths:
            for path_path in pathology_paths:
                try:
                    # 这里简化处理，实际应用中需要切patch
                    img = Image.open(path_path).convert('RGB')
                    if 'pathology' in self.transforms:
                        patch_tensor = self.transforms['pathology'](img)
                        pathology_patches.append(patch_tensor)
                except Exception as e:
                    warnings.warn(f"加载病理切片失败 {path_path}: {e}")
        
        # 构建批次数据
        return self._build_batch_dict(
            texts=[processed_text],
            images_list=[images],
            pathology_list=[pathology_patches]
        )
    
    def _prepare_batch_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """准备批量样本的输入数据"""
        texts = []
        images_list = []
        pathology_list = []
        
        for sample in samples:
            # 文本
            text = sample.get('text', "")
            if 'text' in self.transforms:
                text = self.transforms['text'](text)
            texts.append(text)
            
            # 图片
            images = []
            for img_path in sample.get('image_paths', []):
                try:
                    img = Image.open(img_path).convert('RGB')
                    if 'photo' in self.transforms:
                        img_tensor = self.transforms['photo'](img)
                        images.append(img_tensor)
                except:
                    continue
            images_list.append(images)
            
            # 病理
            patches = []
            for path_path in sample.get('pathology_paths', []):
                try:
                    img = Image.open(path_path).convert('RGB')
                    if 'pathology' in self.transforms:
                        patch_tensor = self.transforms['pathology'](img)
                        patches.append(patch_tensor)
                except:
                    continue
            pathology_list.append(patches)
        
        return self._build_batch_dict(texts, images_list, pathology_list)
    
    def _build_batch_dict(self, 
                         texts: List[str],
                         images_list: List[List[torch.Tensor]],
                         pathology_list: List[List[torch.Tensor]]) -> Dict[str, Any]:
        """构建批次数据字典"""
        batch_size = len(texts)
        
        # 文本数据
        text_data = {
            'texts': texts,
            'lengths': torch.tensor([len(text) for text in texts])
        }
        
        # 图片数据
        max_images = max([len(imgs) for imgs in images_list]) if images_list else 0
        max_images = max(1, max_images)  # 至少为1
        
        if max_images > 0:
            padded_images = torch.zeros(batch_size, max_images, 3, 224, 224)
            image_masks = torch.zeros(batch_size, max_images, dtype=torch.bool)
            image_counts = []
            
            for i, images in enumerate(images_list):
                count = len(images)
                image_counts.append(count)
                if count > 0:
                    imgs_tensor = torch.stack(images[:max_images])
                    actual_count = min(count, max_images)
                    padded_images[i, :actual_count] = imgs_tensor[:actual_count]
                    image_masks[i, :actual_count] = True
        else:
            padded_images = torch.zeros(batch_size, 1, 3, 224, 224)
            image_masks = torch.zeros(batch_size, 1, dtype=torch.bool)
            image_counts = [0] * batch_size
        
        photo_data = {
            'images': padded_images.to(self.device),
            'masks': image_masks.to(self.device),
            'counts': torch.tensor(image_counts)
        }
        
        # 病理数据
        max_patches = max([len(patches) for patches in pathology_list]) if pathology_list else 0
        max_patches = max(1, max_patches)
        
        if max_patches > 0:
            padded_patches = torch.zeros(batch_size, max_patches, 3, 224, 224)
            patch_masks = torch.zeros(batch_size, max_patches, dtype=torch.bool)
            patch_counts = []
            
            for i, patches in enumerate(pathology_list):
                count = len(patches)
                patch_counts.append(count)
                if count > 0:
                    patches_tensor = torch.stack(patches[:max_patches])
                    actual_count = min(count, max_patches)
                    padded_patches[i, :actual_count] = patches_tensor[:actual_count]
                    patch_masks[i, :actual_count] = True
        else:
            padded_patches = torch.zeros(batch_size, 1, 3, 224, 224)
            patch_masks = torch.zeros(batch_size, 1, dtype=torch.bool)
            patch_counts = [0] * batch_size
        
        pathology_data = {
            'patches': padded_patches.to(self.device),
            'masks': patch_masks.to(self.device),
            'counts': torch.tensor(patch_counts),
            'coordinates': [[] for _ in range(batch_size)]
        }
        
        return {
            'text': text_data,
            'photos': photo_data,
            'pathology': pathology_data
        }
    
    def _split_batch_outputs(self, batch_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将批次输出分割为单个样本输出"""
        batch_size = batch_outputs['predictions'].shape[0]
        outputs = []
        
        for i in range(batch_size):
            sample_output = {
                'predictions': batch_outputs['predictions'][i],
                'modal_availability': batch_outputs.get('modal_availability', [None])[i]
            }
            outputs.append(sample_output)
        
        return outputs
    
    def _process_attention_weights(self, attention_weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """处理注意力权重"""
        processed = {}
        
        for modal_name, weights in attention_weights.items():
            if isinstance(weights, torch.Tensor):
                processed[modal_name] = weights[0].cpu().numpy().tolist()  # 取第一个样本
        
        return processed
    
    def _process_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """处理特征"""
        processed = {}
        
        for feature_name, feature_tensor in features.items():
            if isinstance(feature_tensor, torch.Tensor):
                processed[feature_name] = {
                    'shape': list(feature_tensor.shape),
                    'norm': torch.norm(feature_tensor[0]).item()  # 特征向量的L2范数
                }
        
        return processed
    
    def visualize_prediction(self, 
                           result: Dict[str, Any], 
                           save_path: Optional[str] = None,
                           show_probabilities: bool = True) -> Optional[plt.Figure]:
        """
        可视化预测结果
        
        Args:
            result: 预测结果字典
            save_path: 保存路径（可选）
            show_probabilities: 是否显示所有类别概率
            
        Returns:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 预测结果饼图
        ax1 = axes[0]
        predicted_class = result['predicted_class_name']
        confidence = result['confidence']
        
        # 创建饼图数据
        sizes = [confidence, 1 - confidence]
        labels = [f'{predicted_class}\n({confidence:.1%})', f'其他\n({1-confidence:.1%})']
        colors = ['#ff9999', '#66b3ff']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
        ax1.set_title(f'预测结果: {predicted_class}', fontsize=14, fontweight='bold')
        
        # 所有类别概率柱状图
        if show_probabilities:
            ax2 = axes[1]
            class_names = list(result['class_probabilities'].keys())
            probabilities = list(result['class_probabilities'].values())
            
            bars = ax2.bar(range(len(class_names)), probabilities)
            ax2.set_xlabel('类别')
            ax2.set_ylabel('预测概率')
            ax2.set_title('所有类别预测概率')
            ax2.set_xticks(range(len(class_names)))
            ax2.set_xticklabels(class_names, rotation=45, ha='right')
            
            # 高亮预测类别
            max_idx = probabilities.index(max(probabilities))
            bars[max_idx].set_color('#ff9999')
            
            # 添加数值标签
            for i, prob in enumerate(probabilities):
                ax2.text(i, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        
        return fig
    
    def generate_report(self, 
                       result: Dict[str, Any], 
                       sample_info: Optional[Dict[str, Any]] = None) -> str:
        """
        生成预测报告
        
        Args:
            result: 预测结果
            sample_info: 样本信息（可选）
            
        Returns:
            报告文本
        """
        report = []
        report.append("=" * 50)
        report.append("多模态医学图像分类预测报告")
        report.append("=" * 50)
        
        if sample_info:
            report.append(f"\n样本ID: {sample_info.get('id', 'N/A')}")
            report.append(f"患者信息: {sample_info.get('patient_info', 'N/A')}")
        
        report.append(f"\n预测结果:")
        report.append(f"  预测类别: {result['predicted_class_name']}")
        report.append(f"  置信度: {result['confidence']:.4f}")
        
        report.append(f"\n模态可用性:")
        modal_names = ['文本', '口腔照片', '病理切片']
        modal_availability = result['modal_availability']
        for i, (modal_name, available) in enumerate(zip(modal_names, modal_availability)):
            status = "✓ 可用" if available else "✗ 不可用"
            report.append(f"  {modal_name}: {status}")
        
        report.append(f"\n所有类别概率:")
        for class_name, prob in result['class_probabilities'].items():
            report.append(f"  {class_name}: {prob:.4f}")
        
        if 'attention_weights' in result:
            report.append(f"\n注意力权重:")
            for modal_name, weights in result['attention_weights'].items():
                if isinstance(weights, list) and len(weights) > 0:
                    avg_weight = np.mean(weights)
                    report.append(f"  {modal_name}: 平均权重 {avg_weight:.4f}")
        
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='多模态医学图像分类推理')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--config_path', type=str,
                       help='配置文件路径')
    parser.add_argument('--text', type=str, default="",
                       help='病历文本')
    parser.add_argument('--images', type=str, nargs='*',
                       help='口腔照片路径列表')
    parser.add_argument('--pathology', type=str, nargs='*',
                       help='病理切片路径列表')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化结果')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='推理设备')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化推理器
    inferencer = MultiModalInferencer(
        model_path=args.model_path,
        device=args.device,
        config_path=args.config_path
    )
    
    # 执行预测
    result = inferencer.predict_single(
        text=args.text,
        image_paths=args.images or [],
        pathology_paths=args.pathology or [],
        return_attention=True,
        return_features=True
    )
    
    # 生成报告
    report = inferencer.generate_report(result)
    print(report)
    
    # 保存结果
    result_file = output_dir / 'prediction_result.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        # 处理numpy类型以便JSON序列化
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            else:
                serializable_result[key] = value
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {result_file}")
    
    # 生成可视化（如果请求）
    if args.visualize:
        fig = inferencer.visualize_prediction(result)
        if fig:
            viz_file = output_dir / 'prediction_visualization.png'
            fig.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"可视化结果已保存到: {viz_file}")


if __name__ == "__main__":
    main()
