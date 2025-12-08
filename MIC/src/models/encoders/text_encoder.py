"""
文本编码器模块
支持Qwen3等预训练模型进行文本特征提取
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# 关闭transformers警告
logging.set_verbosity_error()


class TextEncoder(nn.Module):
    """
    文本编码器
    支持多种预训练模型: Qwen, BERT, RoBERTa等
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # 模型配置
        self.model_name = config.get("model_name", "bert-base-chinese")
        self.embedding_dim = config.get("embedding_dim", 768)
        self.max_length = config.get("max_length", 512)
        self.freeze_encoder = config.get("freeze_encoder", False)

        # 加载tokenizer和model
        self._load_model()

        # 特征投影层
        if self.model.config.hidden_size != self.embedding_dim:
            self.projection = nn.Linear(
                self.model.config.hidden_size, self.embedding_dim
            )
        else:
            self.projection = nn.Identity()

        # Dropout层
        self.dropout = nn.Dropout(config.get("dropout", 0.1))

        print(f"文本编码器初始化完成:")
        print(f"  模型: {self.model_name}")
        print(f"  输出维度: {self.embedding_dim}")
        print(f"  最大长度: {self.max_length}")
        print(f"  冻结编码器: {self.freeze_encoder}")

    def _load_model(self):
        """加载预训练模型"""
        from pathlib import Path

        # 检查是否是本地路径
        if isinstance(self.model_name, str) and (
            self.model_name.startswith("/")
            or self.model_name.startswith("./")
            or Path(self.model_name).exists()
        ):
            print(f"检测到本地模型路径: {self.model_name}")

            # 验证本地路径是否有效
            model_path = Path(self.model_name)
            if not model_path.exists():
                print(f"警告: 本地路径不存在: {model_path}")
                self._load_fallback_model()
                return

            # 检查必需文件是否存在
            required_files = ["config.json"]
            missing_files = [f for f in required_files if not (model_path / f).exists()]
            if missing_files:
                print(f"警告: 缺少必需文件: {missing_files}")
                print("尝试继续加载...")

        try:
            # 尝试加载指定模型（本地路径或模型名）
            print(f"正在加载模型: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=(
                    Path(self.model_name).exists()
                    if isinstance(self.model_name, str)
                    else False
                ),
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=(
                    Path(self.model_name).exists()
                    if isinstance(self.model_name, str)
                    else False
                ),
            )

            # 添加特殊token（如果需要）
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"成功加载文本编码器: {self.model_name}")

        except Exception as e:
            print(f"警告: 无法加载模型 {self.model_name}: {e}")
            self._load_fallback_model()

        # 冻结编码器参数（如果需要）
        if self.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            print("已冻结预训练模型参数")

    def _load_fallback_model(self):
        """加载回退模型"""
        print("回退到 bert-base-chinese")

        try:
            # 回退到BERT中文模型
            self.model_name = "bert-base-chinese"
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            print("成功加载回退模型: bert-base-chinese")
        except Exception as e:
            print(f"回退模型加载也失败: {e}")
            # 最后的应急方案 - 使用一个更通用的模型
            try:
                self.model_name = "distilbert-base-uncased"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                print("使用应急模型: distilbert-base-uncased")
            except Exception as final_e:
                raise RuntimeError(f"无法加载任何文本编码器模型: {final_e}")

    def forward(
        self, texts: List[str], return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            texts: 文本列表
            return_attention: 是否返回注意力权重

        Returns:
            包含编码特征的字典
        """
        batch_size = len(texts)
        device = next(self.parameters()).device

        # 文本预处理和tokenization
        encoding = self._tokenize_texts(texts)

        # 移动到设备
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # 前向传播
        with torch.set_grad_enabled(not self.freeze_encoder or self.training):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=return_attention,
                return_dict=True,
            )

        # 提取特征
        last_hidden_state = (
            outputs.last_hidden_state
        )  # [batch_size, seq_len, hidden_size]

        # 池化策略：使用[CLS] token或平均池化
        if self._has_cls_token():
            # 使用[CLS] token
            pooled_features = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        else:
            # 加权平均池化（根据attention mask）
            pooled_features = self._mean_pooling(last_hidden_state, attention_mask)

        # 特征投影
        features = self.projection(pooled_features)  # [batch_size, embedding_dim]
        features = self.dropout(features)

        result = {
            "features": features,
            "attention_mask": attention_mask,
            "sequence_output": last_hidden_state,
        }

        # 添加注意力权重（如果需要）
        if return_attention and outputs.attentions is not None:
            result["attentions"] = outputs.attentions

        return result

    def _tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """文本tokenization"""
        # 预处理文本
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                # 空文本处理
                processed_texts.append("[EMPTY]")
            else:
                processed_texts.append(text.strip())

        # Tokenization
        encoding = self.tokenizer(
            processed_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return encoding

    def _has_cls_token(self) -> bool:
        """检查模型是否有[CLS] token"""
        return (
            hasattr(self.tokenizer, "cls_token")
            and self.tokenizer.cls_token is not None
        )

    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """加权平均池化"""
        # 扩展attention mask维度
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        # 加权求和
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """编码文本（推理时使用）"""
        self.eval()
        with torch.no_grad():
            result = self.forward(texts)
            return result["features"]

    def get_text_attention_weights(
        self, texts: List[str]
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        """获取文本注意力权重"""
        self.eval()
        with torch.no_grad():
            result = self.forward(texts, return_attention=True)

            # 处理tokens
            encoding = self._tokenize_texts(texts)
            tokens_list = []
            for i in range(len(texts)):
                tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][i])
                tokens_list.append(tokens)

            attentions = result.get("attentions", None)
            return attentions, tokens_list


class MedicalTextEncoder(TextEncoder):
    """
    医学文本编码器
    针对医学文本进行特殊优化
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        # 医学词汇增强
        self.medical_vocab_file = config.get("medical_vocab_file", None)
        if self.medical_vocab_file:
            self._load_medical_vocabulary()

        # 医学实体识别层（可选）
        if config.get("use_medical_ner", False):
            self.medical_ner = self._create_medical_ner_layer()

    def _load_medical_vocabulary(self):
        """加载医学词汇表"""
        try:
            with open(self.medical_vocab_file, "r", encoding="utf-8") as f:
                medical_terms = [line.strip() for line in f if line.strip()]

            # 添加到tokenizer（如果支持）
            if hasattr(self.tokenizer, "add_tokens"):
                new_tokens = []
                for term in medical_terms:
                    if term not in self.tokenizer.vocab:
                        new_tokens.append(term)

                if new_tokens:
                    self.tokenizer.add_tokens(new_tokens)
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    print(f"添加了 {len(new_tokens)} 个医学术语到词汇表")

        except Exception as e:
            print(f"警告: 加载医学词汇表失败: {e}")

    def _create_medical_ner_layer(self) -> nn.Module:
        """创建医学实体识别层"""
        # 简化版NER层
        return nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, 10),  # 假设10种医学实体类型
        )

    def forward(
        self, texts: List[str], return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """医学文本前向传播"""
        result = super().forward(texts, return_attention)

        # 添加医学实体识别（如果启用）
        if hasattr(self, "medical_ner"):
            ner_logits = self.medical_ner(result["features"])
            result["medical_entities"] = ner_logits

        return result


def create_text_encoder(config: Dict) -> TextEncoder:
    """创建文本编码器工厂函数"""
    encoder_type = config.get("encoder_type", "standard")

    if encoder_type == "medical":
        return MedicalTextEncoder(config)
    else:
        return TextEncoder(config)


if __name__ == "__main__":
    # 测试文本编码器
    config = {
        "model_name": "bert-base-chinese",
        "embedding_dim": 768,
        "max_length": 256,
        "freeze_encoder": False,
    }

    # 创建编码器
    encoder = create_text_encoder(config)

    # 测试文本
    test_texts = [
        "患者口腔内出现红肿，疼痛明显。",
        "建议进行进一步的病理检查。",
        "",  # 空文本测试
    ]

    # 编码测试
    result = encoder(test_texts)

    print("文本编码结果:")
    print(f"  特征形状: {result['features'].shape}")
    print(f"  注意力掩码形状: {result['attention_mask'].shape}")

    # 测试推理模式
    features = encoder.encode_texts(test_texts[:2])
    print(f"  推理特征形状: {features.shape}")

    print("文本编码器测试完成!")
