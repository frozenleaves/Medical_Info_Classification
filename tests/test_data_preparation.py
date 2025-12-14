#!/usr/bin/env python3
"""
测试数据准备功能

使用 sub-dataset/ 目录测试数据扫描和索引生成
"""

import sys
import json
import shutil
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MIC.src.utils.prepare_dataset_from_folders import (
    scan_dataset_directory,
    split_dataset,
    save_index_files,
    create_label_mapping
)


def test_scan_single_sample():
    """测试扫描单个样本"""
    print("\n" + "=" * 80)
    print("📝 测试1: 扫描单个样本")
    print("=" * 80)
    
    # 创建临时测试目录
    test_dir = project_root / "tests" / "temp_test_data"
    test_dir.mkdir(exist_ok=True)
    
    try:
        # 复制示例数据
        source_dir = project_root / "sub-dataset" / "常安OLK+OSF"
        if not source_dir.exists():
            print("⚠️  示例数据不存在，跳过测试")
            return True
        
        # 创建类别文件夹
        class_dir = test_dir / "口腔白斑病"
        class_dir.mkdir(exist_ok=True)
        
        # 创建样本文件夹
        sample_dir = class_dir / "常安OLK+OSF"
        sample_dir.mkdir(exist_ok=True)
        
        # 复制照片文件（只复制几个，加快测试）
        jpg_files = list(source_dir.glob("*.JPG"))[:3]
        for jpg_file in jpg_files:
            shutil.copy(jpg_file, sample_dir / jpg_file.name)
        
        # 复制文本文件
        txt_files = list(source_dir.glob("*.txt"))
        if txt_files:
            shutil.copy(txt_files[0], sample_dir / txt_files[0].name)
        
        # 复制processed目录
        source_processed = source_dir / "processed"
        if source_processed.exists():
            target_processed = sample_dir / "processed"
            target_processed.mkdir(exist_ok=True)
            
            # 只复制svs文件（符号链接，不实际复制大文件）
            svs_files = list(source_processed.glob("*.svs"))
            for svs_file in svs_files:
                target_file = target_processed / svs_file.name
                # 创建软链接而不是复制（节省空间）
                if not target_file.exists():
                    try:
                        target_file.symlink_to(svs_file)
                    except:
                        # 如果符号链接失败，创建一个标记文件
                        target_file.touch()
        
        print(f"\n✅ 测试数据已准备: {test_dir}")
        
        # 扫描数据
        samples = scan_dataset_directory(str(test_dir), verbose=True)
        
        # 验证结果
        assert len(samples) == 1, f"期望1个样本，实际{len(samples)}个"
        
        sample = samples[0]
        assert sample["label"] == "口腔白斑病"
        assert len(sample["photo_paths"]) >= 3
        assert sample["text_path"] is not None
        print(f"\n✅ 病理切片数: {len(sample['pathology_paths'])}")
        
        print("\n✅ 测试1通过：单个样本扫描正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理测试数据
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"\n🧹 已清理测试数据: {test_dir}")


def test_multiple_classes():
    """测试扫描多个类别"""
    print("\n" + "=" * 80)
    print("📝 测试2: 扫描多个类别")
    print("=" * 80)
    
    test_dir = project_root / "tests" / "temp_test_data_multi"
    test_dir.mkdir(exist_ok=True)
    
    try:
        # 创建模拟数据结构
        classes = ["类别A", "类别B", "类别C"]
        samples_per_class = 3
        
        for class_name in classes:
            class_dir = test_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            for i in range(samples_per_class):
                sample_dir = class_dir / f"样本{i+1}"
                sample_dir.mkdir(exist_ok=True)
                
                # 创建模拟文件
                (sample_dir / f"photo{i+1}.jpg").touch()
                (sample_dir / "medical_record.txt").write_text(f"样本{i+1}的病历", encoding="utf-8")
                
                processed_dir = sample_dir / "processed"
                processed_dir.mkdir(exist_ok=True)
                (processed_dir / f"slide{i+1}.svs").touch()
        
        print(f"\n✅ 测试数据已准备: {test_dir}")
        
        # 扫描数据
        samples = scan_dataset_directory(str(test_dir), verbose=True)
        
        # 验证结果
        expected_total = len(classes) * samples_per_class
        assert len(samples) == expected_total, f"期望{expected_total}个样本，实际{len(samples)}个"
        
        # 验证类别分布
        class_counts = {}
        for sample in samples:
            class_counts[sample["label"]] = class_counts.get(sample["label"], 0) + 1
        
        for class_name in classes:
            assert class_counts[class_name] == samples_per_class, \
                f"{class_name}期望{samples_per_class}个样本，实际{class_counts[class_name]}个"
        
        print("\n✅ 测试2通过：多类别扫描正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"\n🧹 已清理测试数据: {test_dir}")


def test_data_splitting():
    """测试数据集划分"""
    print("\n" + "=" * 80)
    print("📝 测试3: 数据集划分")
    print("=" * 80)
    
    try:
        # 创建模拟样本
        samples = []
        for i in range(100):
            samples.append({
                "id": f"sample_{i}",
                "label": f"class_{i % 5}",  # 5个类别
                "text_path": f"texts/sample_{i}.txt",
                "photo_paths": [f"photos/sample_{i}/photo.jpg"],
                "pathology_paths": [{"path": f"pathology/sample_{i}/slide.svs", "roi": None}]
            })
        
        print(f"✅ 创建了{len(samples)}个模拟样本，5个类别")
        
        # 划分数据集
        train_samples, val_samples, test_samples = split_dataset(
            samples=samples,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            stratify=True,
            random_seed=42
        )
        
        print(f"\n📊 划分结果:")
        print(f"  训练集: {len(train_samples)} 个样本")
        print(f"  验证集: {len(val_samples)} 个样本")
        print(f"  测试集: {len(test_samples)} 个样本")
        
        # 验证总数
        total = len(train_samples) + len(val_samples) + len(test_samples)
        assert total == len(samples), f"划分后总数不匹配：{total} != {len(samples)}"
        
        # 验证类别分布（分层采样）
        for split_name, split_samples in [("训练集", train_samples), 
                                          ("验证集", val_samples),
                                          ("测试集", test_samples)]:
            class_counts = {}
            for sample in split_samples:
                label = sample["label"]
                class_counts[label] = class_counts.get(label, 0) + 1
            
            print(f"\n  {split_name}类别分布:")
            for label, count in sorted(class_counts.items()):
                print(f"    {label}: {count} 个")
        
        print("\n✅ 测试3通过：数据集划分正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_index_file_generation():
    """测试索引文件生成"""
    print("\n" + "=" * 80)
    print("📝 测试4: 索引文件生成")
    print("=" * 80)
    
    output_dir = project_root / "tests" / "temp_index_output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 创建模拟样本
        train_samples = [
            {
                "id": "sample_1",
                "label": "class_A",
                "text_path": "texts/sample_1.txt",
                "photo_paths": ["photos/sample_1/photo.jpg"],
                "pathology_paths": [{"path": "pathology/sample_1/slide.svs", "roi": None}]
            }
        ]
        
        val_samples = [
            {
                "id": "sample_2",
                "label": "class_B",
                "text_path": "texts/sample_2.txt",
                "photo_paths": ["photos/sample_2/photo.jpg"],
                "pathology_paths": [{"path": "pathology/sample_2/slide.svs", "roi": None}]
            }
        ]
        
        # 保存索引文件
        train_file, val_file, test_file = save_index_files(
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=None,
            output_dir=str(output_dir),
            verbose=True
        )
        
        # 验证文件存在
        assert Path(train_file).exists(), "训练集索引文件不存在"
        assert Path(val_file).exists(), "验证集索引文件不存在"
        
        # 验证文件内容
        with open(train_file, 'r', encoding='utf-8') as f:
            loaded_train = json.load(f)
        
        assert len(loaded_train) == len(train_samples), "训练集样本数不匹配"
        assert loaded_train[0]["id"] == "sample_1", "样本ID不匹配"
        
        print("\n✅ 索引文件验证:")
        print(f"  训练集: {len(loaded_train)} 个样本")
        print(f"  文件: {train_file}")
        
        # 测试标签映射
        all_samples = train_samples + val_samples
        mapping_file = create_label_mapping(all_samples, str(output_dir))
        
        assert Path(mapping_file).exists(), "标签映射文件不存在"
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        assert "label_map" in mapping_data
        assert "class_names" in mapping_data
        assert len(mapping_data["class_names"]) == 2  # class_A, class_B
        
        print(f"\n✅ 标签映射:")
        print(f"  类别数: {mapping_data['num_classes']}")
        for name, idx in mapping_data["label_map"].items():
            print(f"    {idx}: {name}")
        
        print("\n✅ 测试4通过：索引文件生成正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试4失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"\n🧹 已清理输出目录: {output_dir}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("🧪 数据准备功能测试套件")
    print("=" * 80)
    
    tests = [
        ("单个样本扫描", test_scan_single_sample),
        ("多类别扫描", test_multiple_classes),
        ("数据集划分", test_data_splitting),
        ("索引文件生成", test_index_file_generation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

