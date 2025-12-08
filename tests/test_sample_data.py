#!/usr/bin/env python3
"""
样本数据验证测试（无需外部依赖）
验证A_Datasets目录的数据结构和文件完整性
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SampleDataValidator:
    """样本数据验证器"""

    def __init__(self):
        self.data_dir = project_root / "A_Datasets"
        self.results = {}

    def test_directory_structure(self):
        """测试目录结构"""
        print("📁 测试目录结构...")
        print("-" * 30)

        try:
            # 检查主目录
            if not self.data_dir.exists():
                print(f"❌ A_Datasets目录不存在: {self.data_dir}")
                return False

            print(f"✅ 主数据目录存在: {self.data_dir}")

            # 检查子目录
            required_dirs = ["texts", "photos", "pathology"]
            for dir_name in required_dirs:
                dir_path = self.data_dir / dir_name
                if dir_path.exists():
                    print(f"✅ {dir_name}目录存在")
                else:
                    print(f"❌ {dir_name}目录不存在")
                    return False

            return True

        except Exception as e:
            print(f"❌ 目录结构检查失败: {e}")
            return False

    def test_labels_file(self):
        """测试标签文件"""
        print("\n🏷️ 测试标签文件...")
        print("-" * 30)

        try:
            labels_file = self.data_dir / "labels.json"

            if not labels_file.exists():
                print(f"❌ 标签文件不存在: {labels_file}")
                return False

            # 读取并验证JSON格式
            with open(labels_file, "r", encoding="utf-8") as f:
                labels = json.load(f)

            print(f"✅ 标签文件格式正确")
            print(f"✅ 标签数量: {len(labels)}")

            # 打印标签内容
            for patient_id, label in labels.items():
                print(f"  📋 {patient_id} -> {label}")

            # 检查标签格式
            if not isinstance(labels, dict):
                print("❌ 标签格式错误：应为字典格式")
                return False

            if len(labels) == 0:
                print("❌ 标签文件为空")
                return False

            return True

        except json.JSONDecodeError as e:
            print(f"❌ 标签文件JSON格式错误: {e}")
            return False
        except Exception as e:
            print(f"❌ 标签文件检查失败: {e}")
            return False

    def test_text_files(self):
        """测试文本文件"""
        print("\n📝 测试文本文件...")
        print("-" * 30)

        try:
            texts_dir = self.data_dir / "texts"
            if not texts_dir.exists():
                print("❌ texts目录不存在")
                return False

            # 获取所有.txt文件
            text_files = list(texts_dir.glob("*.txt"))

            if not text_files:
                print("❌ 未找到任何文本文件")
                return False

            print(f"✅ 找到 {len(text_files)} 个文本文件")

            valid_files = 0
            for text_file in text_files:
                try:
                    with open(text_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()

                    if content:
                        file_size = text_file.stat().st_size
                        char_count = len(content)
                        print(
                            f"  ✅ {text_file.name}: {file_size} bytes, {char_count} 字符"
                        )

                        # 显示内容预览
                        preview = (
                            content[:100] + "..." if len(content) > 100 else content
                        )
                        print(f"     预览: {preview}")

                        valid_files += 1
                    else:
                        print(f"  ❌ {text_file.name}: 文件为空")

                except Exception as e:
                    print(f"  ❌ {text_file.name}: 读取失败 - {e}")

            print(f"✅ 有效文本文件: {valid_files}/{len(text_files)}")
            return valid_files > 0

        except Exception as e:
            print(f"❌ 文本文件检查失败: {e}")
            return False

    def test_image_files(self):
        """测试图像文件（不实际加载）"""
        print("\n🖼️ 测试图像文件...")
        print("-" * 30)

        try:
            # 检查照片目录
            photos_dir = self.data_dir / "photos"
            photo_count = 0

            if photos_dir.exists():
                for patient_dir in photos_dir.iterdir():
                    if patient_dir.is_dir():
                        patient_photos = list(patient_dir.glob("*.png")) + list(
                            patient_dir.glob("*.jpg")
                        )
                        photo_count += len(patient_photos)
                        print(f"  📸 {patient_dir.name}: {len(patient_photos)} 张照片")

                        for photo in patient_photos:
                            file_size = photo.stat().st_size
                            print(f"    - {photo.name}: {file_size} bytes")

            # 检查病理目录
            pathology_dir = self.data_dir / "pathology"
            pathology_count = 0

            if pathology_dir.exists():
                for patient_dir in pathology_dir.iterdir():
                    if patient_dir.is_dir():
                        patient_slides = list(patient_dir.glob("*.tif")) + list(
                            patient_dir.glob("*.tiff")
                        )
                        pathology_count += len(patient_slides)
                        print(
                            f"  🔬 {patient_dir.name}: {len(patient_slides)} 张病理切片"
                        )

                        for slide in patient_slides:
                            file_size = slide.stat().st_size
                            print(f"    - {slide.name}: {file_size} bytes")

            print(
                f"✅ 总计图像文件: 照片 {photo_count} 张, 病理切片 {pathology_count} 张"
            )
            return (photo_count + pathology_count) > 0

        except Exception as e:
            print(f"❌ 图像文件检查失败: {e}")
            return False

    def test_data_consistency(self):
        """测试数据一致性"""
        print("\n🔍 测试数据一致性...")
        print("-" * 30)

        try:
            # 读取标签文件
            labels_file = self.data_dir / "labels.json"
            if not labels_file.exists():
                print("❌ 无法进行一致性检查：标签文件不存在")
                return False

            with open(labels_file, "r", encoding="utf-8") as f:
                labels = json.load(f)

            consistency_results = {}

            for patient_id in labels.keys():
                patient_data = {
                    "has_text": False,
                    "has_photos": False,
                    "has_pathology": False,
                }

                # 检查文本文件
                text_file = self.data_dir / "texts" / f"{patient_id}.txt"
                patient_data["has_text"] = text_file.exists()

                # 检查照片文件
                photo_dir = self.data_dir / "photos" / patient_id
                if photo_dir.exists():
                    photos = list(photo_dir.glob("*.png")) + list(
                        photo_dir.glob("*.jpg")
                    )
                    patient_data["has_photos"] = len(photos) > 0

                # 检查病理文件
                pathology_dir = self.data_dir / "pathology" / patient_id
                if pathology_dir.exists():
                    slides = list(pathology_dir.glob("*.tif")) + list(
                        pathology_dir.glob("*.tiff")
                    )
                    patient_data["has_pathology"] = len(slides) > 0

                consistency_results[patient_id] = patient_data

            # 打印一致性结果
            print("患者数据完整性检查:")
            complete_patients = 0

            for patient_id, data in consistency_results.items():
                text_status = "✅" if data["has_text"] else "❌"
                photo_status = "✅" if data["has_photos"] else "❌"
                pathology_status = "✅" if data["has_pathology"] else "❌"

                is_complete = all(data.values())
                complete_patients += 1 if is_complete else 0

                completeness = "完整" if is_complete else "不完整"
                print(
                    f"  {patient_id}: 文本{text_status} 照片{photo_status} 病理{pathology_status} ({completeness})"
                )

            total_patients = len(consistency_results)
            completeness_rate = (
                (complete_patients / total_patients * 100) if total_patients > 0 else 0
            )

            print(f"\n📊 数据完整性统计:")
            print(f"  总样本数: {total_patients}")
            print(f"  完整数据样本: {complete_patients}")
            print(f"  完整性率: {completeness_rate:.1f}%")

            return complete_patients > 0

        except Exception as e:
            print(f"❌ 数据一致性检查失败: {e}")
            return False

    def generate_data_report(self):
        """生成数据报告"""
        print("\n📋 数据集报告...")
        print("-" * 30)

        try:
            report = {
                "data_directory": str(self.data_dir),
                "timestamp": str(Path(__file__).stat().st_mtime),
                "structure": {},
                "files": {},
                "statistics": {},
            }

            # 统计文件数量
            if (self.data_dir / "texts").exists():
                text_files = list((self.data_dir / "texts").glob("*.txt"))
                report["files"]["texts"] = len(text_files)

            if (self.data_dir / "photos").exists():
                photo_count = 0
                for patient_dir in (self.data_dir / "photos").iterdir():
                    if patient_dir.is_dir():
                        photos = list(patient_dir.glob("*.png")) + list(
                            patient_dir.glob("*.jpg")
                        )
                        photo_count += len(photos)
                report["files"]["photos"] = photo_count

            if (self.data_dir / "pathology").exists():
                pathology_count = 0
                for patient_dir in (self.data_dir / "pathology").iterdir():
                    if patient_dir.is_dir():
                        slides = list(patient_dir.glob("*.tif")) + list(
                            patient_dir.glob("*.tiff")
                        )
                        pathology_count += len(slides)
                report["files"]["pathology"] = pathology_count

            # 读取标签统计
            labels_file = self.data_dir / "labels.json"
            if labels_file.exists():
                with open(labels_file, "r", encoding="utf-8") as f:
                    labels = json.load(f)

                report["statistics"]["total_patients"] = len(labels)

                # 统计类别分布
                class_distribution = {}
                for label in labels.values():
                    class_distribution[label] = class_distribution.get(label, 0) + 1
                report["statistics"]["class_distribution"] = class_distribution

            # 打印报告
            print("🔢 文件统计:")
            for file_type, count in report.get("files", {}).items():
                print(f"  {file_type}: {count} 个")

            print("\n📊 样本统计:")
            stats = report.get("statistics", {})
            print(f"  总样本数: {stats.get('total_patients', 0)}")

            print("\n🏷️ 类别分布:")
            for class_name, count in stats.get("class_distribution", {}).items():
                print(f"  {class_name}: {count} 例")

            print(
                f"\n✅ 数据集概要: 这是一个包含 {stats.get('total_patients', 0)} 个患者样本的多模态医学数据集"
            )

            return True

        except Exception as e:
            print(f"❌ 报告生成失败: {e}")
            return False

    def run_all_tests(self):
        """运行所有验证测试"""
        print("🔬 开始样本数据验证测试...")
        print("=" * 50)

        tests = [
            ("目录结构检查", self.test_directory_structure),
            ("标签文件验证", self.test_labels_file),
            ("文本文件检查", self.test_text_files),
            ("图像文件检查", self.test_image_files),
            ("数据一致性验证", self.test_data_consistency),
            ("数据报告生成", self.generate_data_report),
        ]

        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
                self.results[test_name] = result
            except Exception as e:
                print(f"\n❌ {test_name}出现异常: {e}")
                results.append((test_name, False))
                self.results[test_name] = False

        # 汇总结果
        print("\n" + "=" * 50)
        print("📋 样本数据验证结果:")

        success_count = 0
        for test_name, success in results:
            status = "✅ 通过" if success else "❌ 失败"
            print(f"  {test_name}: {status}")
            if success:
                success_count += 1

        total_tests = len(results)
        print(f"\n🎯 总体结果: {success_count}/{total_tests} 测试通过")

        if success_count == total_tests:
            print("\n🎉 所有验证通过！样本数据完整有效！")
            print("\n✨ 数据集可用于:")
            print("  ✅ 多模态模型训练")
            print("  ✅ 数据加载功能测试")
            print("  ✅ 端到端系统验证")
            return True
        else:
            print(f"\n⚠️ {total_tests - success_count} 项验证失败")
            print("\n📋 建议:")
            print("  1. 检查数据文件完整性")
            print("  2. 确认标签文件格式正确")
            print("  3. 验证图像文件未损坏")
            return False


def main():
    """主函数"""
    validator = SampleDataValidator()
    success = validator.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
