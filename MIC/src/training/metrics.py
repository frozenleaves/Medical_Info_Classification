"""
评估指标模块
实现多种适合医学分类任务的评估指标
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
import warnings


class MetricCalculator:
    """
    评估指标计算器
    支持多种医学分类指标
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        average: str = "macro",
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.average = average  # 'macro', 'micro', 'weighted'

        # 存储预测和真实值
        self.reset()

    def reset(self):
        """重置累积的预测和真实值"""
        self.predictions = []
        self.true_labels = []
        self.probabilities = []

    def update(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ):
        """
        更新累积的预测结果

        Args:
            predictions: [batch_size] 预测类别
            true_labels: [batch_size] 真实标签
            probabilities: [batch_size, num_classes] 预测概率
        """
        self.predictions.extend(predictions.tolist())
        self.true_labels.extend(true_labels.tolist())

        if probabilities is not None:
            self.probabilities.extend(probabilities.tolist())

    def compute(self) -> Dict[str, float]:
        """
        计算所有评估指标

        Returns:
            包含各种指标的字典
        """
        if not self.predictions:
            return {}

        predictions = np.array(self.predictions)
        true_labels = np.array(self.true_labels)

        metrics = {}

        # 基本分类指标
        metrics.update(self._compute_basic_metrics(predictions, true_labels))

        # 混淆矩阵相关指标
        metrics.update(self._compute_confusion_matrix_metrics(predictions, true_labels))

        # 概率相关指标
        if self.probabilities:
            probabilities = np.array(self.probabilities)
            metrics.update(
                self._compute_probability_metrics(true_labels, probabilities)
            )

        # 类别平衡相关指标
        metrics.update(self._compute_balance_metrics(predictions, true_labels))

        return metrics

    def _compute_basic_metrics(
        self, predictions: np.ndarray, true_labels: np.ndarray
    ) -> Dict[str, float]:
        """计算基本分类指标"""
        metrics = {}

        # 准确率
        metrics["accuracy"] = accuracy_score(true_labels, predictions)

        # 精确率、召回率、F1分数
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            metrics["precision"] = precision_score(
                true_labels, predictions, average=self.average, zero_division=0
            )
            metrics["recall"] = recall_score(
                true_labels, predictions, average=self.average, zero_division=0
            )
            metrics["f1_score"] = f1_score(
                true_labels, predictions, average=self.average, zero_division=0
            )

        # 各类别的详细指标
        per_class_precision = precision_score(
            true_labels, predictions, average=None, zero_division=0
        )
        per_class_recall = recall_score(
            true_labels, predictions, average=None, zero_division=0
        )
        per_class_f1 = f1_score(true_labels, predictions, average=None, zero_division=0)

        for i, class_name in enumerate(self.class_names):
            metrics[f"{class_name}_precision"] = (
                per_class_precision[i] if i < len(per_class_precision) else 0.0
            )
            metrics[f"{class_name}_recall"] = (
                per_class_recall[i] if i < len(per_class_recall) else 0.0
            )
            metrics[f"{class_name}_f1"] = (
                per_class_f1[i] if i < len(per_class_f1) else 0.0
            )

        return metrics

    def _compute_confusion_matrix_metrics(
        self, predictions: np.ndarray, true_labels: np.ndarray
    ) -> Dict[str, float]:
        """计算基于混淆矩阵的指标"""
        metrics = {}

        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)

        # 每类的TP, FP, FN, TN
        for i, class_name in enumerate(self.class_names):
            if i < len(cm):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn

                # 敏感性（召回率）
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                metrics[f"{class_name}_sensitivity"] = sensitivity

                # 特异性
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                metrics[f"{class_name}_specificity"] = specificity

                # PPV (正预测值)
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                metrics[f"{class_name}_ppv"] = ppv

                # NPV (负预测值)
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                metrics[f"{class_name}_npv"] = npv

        return metrics

    def _compute_probability_metrics(
        self, true_labels: np.ndarray, probabilities: np.ndarray
    ) -> Dict[str, float]:
        """计算基于概率的指标"""
        metrics = {}

        try:
            # ROC-AUC
            if self.num_classes == 2:
                # 二分类
                auc = roc_auc_score(true_labels, probabilities[:, 1])
                metrics["roc_auc"] = auc
            else:
                # 多分类：one-vs-rest
                auc_scores = []
                for i in range(self.num_classes):
                    try:
                        y_true_binary = (true_labels == i).astype(int)
                        y_score = probabilities[:, i]

                        if len(np.unique(y_true_binary)) > 1:  # 确保有正负样本
                            auc = roc_auc_score(y_true_binary, y_score)
                            auc_scores.append(auc)
                            metrics[f"{self.class_names[i]}_auc"] = auc
                    except:
                        continue

                if auc_scores:
                    metrics["roc_auc_macro"] = np.mean(auc_scores)

            # PR-AUC (Average Precision)
            ap_scores = []
            for i in range(self.num_classes):
                try:
                    y_true_binary = (true_labels == i).astype(int)
                    y_score = probabilities[:, i]

                    if len(np.unique(y_true_binary)) > 1:
                        ap = average_precision_score(y_true_binary, y_score)
                        ap_scores.append(ap)
                        metrics[f"{self.class_names[i]}_ap"] = ap
                except:
                    continue

            if ap_scores:
                metrics["average_precision_macro"] = np.mean(ap_scores)

            # Top-K准确率
            for k in [2, 3]:
                if k < self.num_classes:
                    top_k_acc = self._compute_top_k_accuracy(
                        true_labels, probabilities, k
                    )
                    metrics[f"top_{k}_accuracy"] = top_k_acc

        except Exception as e:
            print(f"警告: 计算概率指标时出错: {e}")

        return metrics

    def _compute_balance_metrics(
        self, predictions: np.ndarray, true_labels: np.ndarray
    ) -> Dict[str, float]:
        """计算类别平衡相关指标"""
        metrics = {}

        # 平衡准确率
        metrics["balanced_accuracy"] = balanced_accuracy_score(true_labels, predictions)

        # Cohen's Kappa
        metrics["cohen_kappa"] = self._compute_cohen_kappa(predictions, true_labels)

        # 类别分布差异
        true_dist = np.bincount(true_labels, minlength=self.num_classes) / len(
            true_labels
        )
        pred_dist = np.bincount(predictions, minlength=self.num_classes) / len(
            predictions
        )

        # KL散度
        kl_div = self._compute_kl_divergence(true_dist, pred_dist)
        metrics["prediction_distribution_kl"] = kl_div

        return metrics

    def _compute_top_k_accuracy(
        self, true_labels: np.ndarray, probabilities: np.ndarray, k: int
    ) -> float:
        """计算Top-K准确率"""
        top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
        correct = 0

        for i, true_label in enumerate(true_labels):
            if true_label in top_k_preds[i]:
                correct += 1

        return correct / len(true_labels)

    def _compute_cohen_kappa(
        self, predictions: np.ndarray, true_labels: np.ndarray
    ) -> float:
        """计算Cohen's Kappa系数"""
        cm = confusion_matrix(true_labels, predictions)
        n = np.sum(cm)

        # 观察到的一致性
        po = np.trace(cm) / n

        # 期望的一致性
        marginal_true = np.sum(cm, axis=1) / n
        marginal_pred = np.sum(cm, axis=0) / n
        pe = np.sum(marginal_true * marginal_pred)

        # Cohen's Kappa
        if pe == 1:
            return 1.0
        else:
            return (po - pe) / (1 - pe)

    def _compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算KL散度"""
        # 避免log(0)
        p = np.clip(p, 1e-8, 1.0)
        q = np.clip(q, 1e-8, 1.0)

        return np.sum(p * np.log(p / q))

    def get_classification_report(self) -> str:
        """获取详细的分类报告"""
        if not self.predictions:
            return "No predictions available"

        predictions = np.array(self.predictions)
        true_labels = np.array(self.true_labels)

        return classification_report(
            true_labels, predictions, target_names=self.class_names, digits=4
        )

    def get_confusion_matrix(self) -> np.ndarray:
        """获取混淆矩阵"""
        if not self.predictions:
            return np.array([])

        predictions = np.array(self.predictions)
        true_labels = np.array(self.true_labels)

        return confusion_matrix(true_labels, predictions)


class MedicalMetrics:
    """
    医学专用评估指标
    """

    @staticmethod
    def compute_diagnostic_metrics(
        tp: int, tn: int, fp: int, fn: int
    ) -> Dict[str, float]:
        """
        计算诊断相关指标

        Args:
            tp: True Positive
            tn: True Negative
            fp: False Positive
            fn: False Negative

        Returns:
            诊断指标字典
        """
        metrics = {}

        # 敏感性 (Sensitivity / Recall / True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["sensitivity"] = sensitivity

        # 特异性 (Specificity / True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["specificity"] = specificity

        # 阳性预测值 (Positive Predictive Value / Precision)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics["ppv"] = ppv

        # 阴性预测值 (Negative Predictive Value)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        metrics["npv"] = npv

        # 准确率 (Accuracy)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        metrics["accuracy"] = accuracy

        # 似然比
        # 阳性似然比
        lr_positive = (
            sensitivity / (1 - specificity) if specificity < 1 else float("inf")
        )
        metrics["lr_positive"] = lr_positive

        # 阴性似然比
        lr_negative = (
            (1 - sensitivity) / specificity if specificity > 0 else float("inf")
        )
        metrics["lr_negative"] = lr_negative

        # 诊断优势比 (Diagnostic Odds Ratio)
        dor = lr_positive / lr_negative if lr_negative > 0 else float("inf")
        metrics["diagnostic_odds_ratio"] = dor

        # Youden指数
        youden_index = sensitivity + specificity - 1
        metrics["youden_index"] = youden_index

        # F1分数
        f1 = (
            2 * (ppv * sensitivity) / (ppv + sensitivity)
            if (ppv + sensitivity) > 0
            else 0.0
        )
        metrics["f1_score"] = f1

        return metrics

    @staticmethod
    def compute_roc_metrics(
        true_labels: np.ndarray, probabilities: np.ndarray
    ) -> Dict[str, float]:
        """
        计算ROC相关指标

        Args:
            true_labels: 真实标签
            probabilities: 预测概率

        Returns:
            ROC指标字典
        """
        from sklearn.metrics import roc_curve, auc

        metrics = {}

        try:
            # 计算ROC曲线
            fpr, tpr, thresholds = roc_curve(true_labels, probabilities)

            # AUC
            auc_score = auc(fpr, tpr)
            metrics["auc"] = auc_score

            # 找到最优阈值（Youden指数最大）
            youden_scores = tpr - fpr
            optimal_idx = np.argmax(youden_scores)

            metrics["optimal_threshold"] = thresholds[optimal_idx]
            metrics["optimal_sensitivity"] = tpr[optimal_idx]
            metrics["optimal_specificity"] = 1 - fpr[optimal_idx]
            metrics["optimal_youden"] = youden_scores[optimal_idx]

        except Exception as e:
            print(f"警告: 计算ROC指标时出错: {e}")

        return metrics


class MultiModalMetrics:
    """
    多模态模型专用评估指标
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        # 主任务指标
        self.main_metrics = MetricCalculator(num_classes, class_names)

        # 各模态辅助任务指标
        self.modal_metrics = {
            "text": MetricCalculator(num_classes, class_names),
            "photo": MetricCalculator(num_classes, class_names),
            "pathology": MetricCalculator(num_classes, class_names),
        }

        # 模态可用性统计
        self.modal_availability_stats = {
            "text_available": 0,
            "photo_available": 0,
            "pathology_available": 0,
            "total_samples": 0,
        }

    def reset(self):
        """重置所有指标"""
        self.main_metrics.reset()
        for metrics in self.modal_metrics.values():
            metrics.reset()

        self.modal_availability_stats = {
            "text_available": 0,
            "photo_available": 0,
            "pathology_available": 0,
            "total_samples": 0,
        }

    def update(
        self,
        outputs: Dict[str, Any],
        targets: np.ndarray,
        modal_availability: Optional[np.ndarray] = None,
    ):
        """
        更新多模态指标

        Args:
            outputs: 模型输出字典
            targets: 真实标签
            modal_availability: 模态可用性掩码
        """
        # 主任务指标
        main_preds = outputs["predictions"].argmax(axis=1)
        main_probs = outputs["predictions"]

        self.main_metrics.update(main_preds, targets, main_probs)

        # 辅助任务指标
        if "auxiliary" in outputs:
            aux_outputs = outputs["auxiliary"]

            for modal_name in ["text", "photo", "pathology"]:
                logits_key = f"{modal_name}_logits"
                if logits_key in aux_outputs:
                    aux_probs = torch.softmax(
                        torch.tensor(aux_outputs[logits_key]), dim=1
                    ).numpy()
                    aux_preds = aux_probs.argmax(axis=1)

                    self.modal_metrics[modal_name].update(aux_preds, targets, aux_probs)

        # 模态可用性统计
        if modal_availability is not None:
            batch_size = len(targets)
            self.modal_availability_stats["total_samples"] += batch_size
            self.modal_availability_stats["text_available"] += modal_availability[
                :, 0
            ].sum()
            self.modal_availability_stats["photo_available"] += modal_availability[
                :, 1
            ].sum()
            self.modal_availability_stats["pathology_available"] += modal_availability[
                :, 2
            ].sum()

    def compute(self) -> Dict[str, Any]:
        """计算所有多模态指标"""
        results = {}

        # 主任务指标
        results["main"] = self.main_metrics.compute()

        # 辅助任务指标
        results["auxiliary"] = {}
        for modal_name, metrics in self.modal_metrics.items():
            modal_results = metrics.compute()
            if modal_results:  # 只有当有数据时才添加
                results["auxiliary"][modal_name] = modal_results

        # 模态可用性统计
        total = self.modal_availability_stats["total_samples"]
        if total > 0:
            results["modal_availability"] = {
                "text_coverage": self.modal_availability_stats["text_available"]
                / total,
                "photo_coverage": self.modal_availability_stats["photo_available"]
                / total,
                "pathology_coverage": self.modal_availability_stats[
                    "pathology_available"
                ]
                / total,
                "total_samples": total,
            }

        return results


if __name__ == "__main__":
    # 测试评估指标
    import torch

    # 创建测试数据
    num_classes = 6
    batch_size = 20
    class_names = [f"Disease_{i}" for i in range(num_classes)]

    # 模拟预测结果
    true_labels = np.random.randint(0, num_classes, batch_size)
    predictions = np.random.randint(0, num_classes, batch_size)
    probabilities = np.random.rand(batch_size, num_classes)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)  # 归一化

    # 测试基本指标计算器
    calculator = MetricCalculator(num_classes, class_names)
    calculator.update(predictions, true_labels, probabilities)

    metrics = calculator.compute()

    print("基本指标测试结果:")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")

    # 测试分类报告
    print("\n分类报告:")
    print(calculator.get_classification_report())

    # 测试医学指标
    print("\n医学诊断指标测试:")
    tp, tn, fp, fn = 15, 25, 5, 10
    med_metrics = MedicalMetrics.compute_diagnostic_metrics(tp, tn, fp, fn)

    for metric_name, value in med_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # 测试多模态指标
    print("\n多模态指标测试:")
    multimodal_metrics = MultiModalMetrics(num_classes, class_names)

    # 模拟模型输出
    model_outputs = {
        "predictions": probabilities,
        "auxiliary": {
            "text_logits": torch.randn(batch_size, num_classes).numpy(),
            "photo_logits": torch.randn(batch_size, num_classes).numpy(),
            "pathology_logits": torch.randn(batch_size, num_classes).numpy(),
        },
    }

    # 模态可用性
    modal_availability = np.ones((batch_size, 3), dtype=bool)
    modal_availability[:5, 2] = False  # 前5个样本没有病理数据

    multimodal_metrics.update(model_outputs, true_labels, modal_availability)

    mm_results = multimodal_metrics.compute()

    print("主任务F1分数:", mm_results["main"].get("f1_score", "N/A"))
    print("模态覆盖率:", mm_results.get("modal_availability", {}))

    print("\n指标计算测试完成!")
