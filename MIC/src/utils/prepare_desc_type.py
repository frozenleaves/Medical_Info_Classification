import re
import json
import random
from typing import List, Dict, Tuple
from pathlib import Path
from collections import defaultdict

class DiseaseDescriptionPairGenerator:
    """专门用于生成病情描述-疾病名称配对的数据生成器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.records = []
        self.disease_to_descriptions = defaultdict(list)
        
    def parse_file(self, filename: str) -> List[Dict]:
        """解析单个txt文件"""
        filepath = self.data_dir / filename
        print(f"正在解析: {filename}")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 按id分割记录
        records = re.split(r'\n\nid=', content)
        parsed_records = []
        
        for i, record in enumerate(records):
            if not record.strip():
                continue
                
            if i == 0:
                record = record.replace('id=', '', 1)
            
            parsed = self.parse_record(record)
            if parsed:
                parsed_records.append(parsed)
        
        return parsed_records
    
    def parse_record(self, record: str) -> Dict:
        """解析单条记录，提取疾病和病情描述"""
        try:
            result = {
                'disease': '',
                'description': '',
                'symptoms': [],
                'duration': '',
                'help_wanted': '',
                'hospital': '',
                'medication': ''
            }
            
            # 提取疾病名称
            disease_match = re.search(r'疾病[：:]\s*\n?(.+?)(?:\n|病情描述|患病时长)', 
                                     record, re.DOTALL)
            if disease_match:
                disease = disease_match.group(1).strip()
                # 清理疾病名称
                disease = re.sub(r'[？?]$', '', disease)  # 去掉问号
                result['disease'] = disease
            
            # 提取主要病情描述
            desc_match = re.search(r'病情描述[：:]\s*\n?(.+?)(?:希望|所就诊|Dialogue|用药情况|过敏史|既往病史|患病时长|$)', 
                                  record, re.DOTALL)
            if desc_match:
                description = desc_match.group(1).strip()
                # 清理描述文本
                description = self.clean_description(description)
                result['description'] = description
            
            # 提取患病时长
            duration_match = re.search(r'患病时长[：:]\s*\n?(.+?)(?:\n|$)', record)
            if duration_match:
                result['duration'] = duration_match.group(1).strip()
            
            # 提取希望获得的帮助
            help_match = re.search(r'希望(?:获得的帮助|提供的帮助)[：:]\s*\n?(.+?)(?:所就诊|用药情况|过敏史|Dialogue|$)', 
                                  record, re.DOTALL)
            if help_match:
                result['help_wanted'] = help_match.group(1).strip()
            
            # 提取就诊医院
            hospital_match = re.search(r'(?:所就诊医院|已就诊医院)(?:科室|及科室)?[：:]\s*\n?(.+?)(?:\n|用药情况|过敏史|Dialogue|$)', 
                                      record, re.DOTALL)
            if hospital_match:
                result['hospital'] = hospital_match.group(1).strip()
            
            # 提取用药情况
            med_match = re.search(r'用药情况[：:]\s*\n?(.+?)(?:过敏史|既往病史|Dialogue|$)', 
                                 record, re.DOTALL)
            if med_match:
                result['medication'] = med_match.group(1).strip()
            
            return result if result['disease'] and result['description'] else None
            
        except Exception as e:
            return None
    
    def clean_description(self, text: str) -> str:
        """清理病情描述文本"""
        # 移除图片标记
        text = re.sub(r'图片|真情寄语', '', text)
        # 移除多余的空白
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。、；：！？（）\[\]【】,.;:!?()\-]', '', text)
        return text.strip()
    
    def load_all_files(self, max_files: int = None):
        """加载所有txt文件"""
        txt_files = sorted([f for f in self.data_dir.glob('*.txt') 
                           if f.name != '.DS_Store'])
        
        if max_files:
            txt_files = txt_files[:max_files]
        
        print(f"找到 {len(txt_files)} 个文件")
        
        for txt_file in txt_files:
            records = self.parse_file(txt_file.name)
            self.records.extend(records)
            
            # 建立疾病到描述的索引
            for record in records:
                if record and record.get('disease'):
                    self.disease_to_descriptions[record['disease']].append(record)
            
            print(f"  - 当前总记录数: {len(self.records)}")
        
        print(f"\n总计解析 {len(self.records)} 条记录")
        print(f"共 {len(self.disease_to_descriptions)} 种不同的疾病")
        return self.records
    
    def create_disease_description_pairs(self, 
                                        output_file: str = "disease_desc_pairs.json",
                                        strategies: List[str] = None) -> List[Dict]:
        """
        创建病情描述-疾病名称配对数据
        
        Args:
            output_file: 输出文件路径
            strategies: 使用的策略列表，可选：
                - 'simple': 简单配对（病情描述 -> 疾病名）
                - 'with_duration': 包含患病时长
                - 'with_symptoms': 提取关键症状
                - 'augmented': 数据增强（同义改写）
                - 'multi_field': 多字段组合
        """
        if strategies is None:
            strategies = ['simple', 'with_duration', 'multi_field']
        
        all_pairs = []
        
        print("\n开始创建病情描述-疾病名称配对...")
        print(f"使用策略: {strategies}")
        
        for record in self.records:
            disease = record.get('disease', '').strip()
            description = record.get('description', '').strip()
            
            if not disease or not description:
                continue
            
            # 策略1: 简单的病情描述-疾病配对
            if 'simple' in strategies:
                pairs = self._create_simple_pairs(record)
                all_pairs.extend(pairs)
            
            # 策略2: 包含患病时长的配对
            if 'with_duration' in strategies:
                pairs = self._create_duration_pairs(record)
                all_pairs.extend(pairs)
            
            # 策略3: 多字段组合配对
            if 'multi_field' in strategies:
                pairs = self._create_multifield_pairs(record)
                all_pairs.extend(pairs)
            
            # 策略4: 数据增强
            if 'augmented' in strategies:
                pairs = self._create_augmented_pairs(record)
                all_pairs.extend(pairs)
        
        # 创建负样本
        print("\n创建负样本...")
        negative_pairs = self._create_negative_pairs(len(all_pairs))
        all_pairs.extend(negative_pairs)
        
        # 打乱数据
        random.shuffle(all_pairs)
        
        # 统计信息
        self._print_statistics(all_pairs)
        
        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_pairs, f, ensure_ascii=False, indent=2)
        
        print(f"\n数据已保存到: {output_file}")
        
        return all_pairs
    
    def _create_simple_pairs(self, record: Dict) -> List[Dict]:
        """策略1: 简单配对"""
        pairs = []
        
        disease = record['disease']
        description = record['description']
        
        # 如果描述太长，截取前500字符
        if len(description) > 500:
            description = description[:500]
        
        pairs.append({
            'text1': description,
            'text2': disease,
            'label': 1,
            'type': 'disease_desc_simple',
            'strategy': 'simple'
        })
        
        return pairs
    
    def _create_duration_pairs(self, record: Dict) -> List[Dict]:
        """策略2: 包含患病时长"""
        pairs = []
        
        disease = record['disease']
        description = record['description']
        duration = record.get('duration', '')
        
        if duration:
            # 组合病情描述和患病时长
            combined_text = f"{description[:400]} 患病时长：{duration}"
            
            pairs.append({
                'text1': combined_text,
                'text2': disease,
                'label': 1,
                'type': 'disease_desc_duration',
                'strategy': 'with_duration'
            })
        
        return pairs
    
    def _create_multifield_pairs(self, record: Dict) -> List[Dict]:
        """策略3: 多字段组合"""
        pairs = []
        
        disease = record['disease']
        
        # 组合多个字段
        fields = []
        if record.get('description'):
            fields.append(f"病情：{record['description'][:300]}")
        if record.get('duration'):
            fields.append(f"患病时长：{record['duration']}")
        if record.get('medication'):
            fields.append(f"用药：{record['medication'][:100]}")
        if record.get('help_wanted'):
            fields.append(f"问题：{record['help_wanted'][:100]}")
        
        if len(fields) >= 2:
            combined_text = ' '.join(fields)
            
            pairs.append({
                'text1': combined_text,
                'text2': disease,
                'label': 1,
                'type': 'disease_desc_multifield',
                'strategy': 'multi_field'
            })
        
        return pairs
    
    def _create_augmented_pairs(self, record: Dict) -> List[Dict]:
        """策略4: 数据增强（同义改写、分段）"""
        pairs = []
        
        disease = record['disease']
        description = record['description']
        
        # 分段处理长文本
        if len(description) > 200:
            # 按句子分割
            sentences = re.split(r'[。！？;；]', description)
            
            # 取前几句作为一个样本
            if len(sentences) >= 2:
                first_part = '。'.join(sentences[:2]) + '。'
                pairs.append({
                    'text1': first_part,
                    'text2': disease,
                    'label': 1,
                    'type': 'disease_desc_segment',
                    'strategy': 'augmented'
                })
            
            # 取中间部分作为另一个样本
            if len(sentences) >= 4:
                middle_part = '。'.join(sentences[2:4]) + '。'
                pairs.append({
                    'text1': middle_part,
                    'text2': disease,
                    'label': 1,
                    'type': 'disease_desc_segment',
                    'strategy': 'augmented'
                })
        
        return pairs
    
    def _create_negative_pairs(self, num_positives: int) -> List[Dict]:
        """创建负样本"""
        negative_pairs = []
        
        # 负样本数量为正样本的一半到相同
        num_negatives = num_positives // 2
        
        print(f"创建 {num_negatives} 个负样本...")
        
        for _ in range(num_negatives):
            # 随机选择两个不同的记录
            if len(self.records) < 2:
                break
                
            r1, r2 = random.sample(self.records, 2)
            
            # 病情描述来自记录1，疾病名来自记录2（不匹配）
            negative_pairs.append({
                'text1': r1['description'][:500],
                'text2': r2['disease'],
                'label': 0,
                'type': 'disease_desc_negative',
                'strategy': 'negative'
            })
        
        return negative_pairs
    
    def _print_statistics(self, pairs: List[Dict]):
        """打印统计信息"""
        print("\n" + "="*50)
        print("数据集统计")
        print("="*50)
        
        total = len(pairs)
        positive = sum(1 for p in pairs if p['label'] == 1)
        negative = sum(1 for p in pairs if p['label'] == 0)
        
        print(f"总配对数: {total}")
        print(f"正样本: {positive} ({positive/total*100:.1f}%)")
        print(f"负样本: {negative} ({negative/total*100:.1f}%)")
        
        # 按策略统计
        strategy_counts = defaultdict(int)
        for p in pairs:
            strategy_counts[p.get('strategy', 'unknown')] += 1
        
        print("\n按策略分布:")
        for strategy, count in sorted(strategy_counts.items()):
            print(f"  {strategy}: {count} ({count/total*100:.1f}%)")
        
        # 按类型统计
        type_counts = defaultdict(int)
        for p in pairs:
            type_counts[p.get('type', 'unknown')] += 1
        
        print("\n按类型分布:")
        for ptype, count in sorted(type_counts.items()):
            print(f"  {ptype}: {count} ({count/total*100:.1f}%)")
    
    def analyze_disease_distribution(self):
        """分析疾病分布"""
        print("\n" + "="*50)
        print("疾病分布分析")
        print("="*50)
        
        # 统计每种疾病的样本数
        disease_counts = defaultdict(int)
        for record in self.records:
            if record.get('disease'):
                disease_counts[record['disease']] += 1
        
        # 排序
        sorted_diseases = sorted(disease_counts.items(), 
                                key=lambda x: x[1], reverse=True)
        
        print(f"\n共有 {len(sorted_diseases)} 种不同的疾病")
        print(f"\n样本数Top 20的疾病:")
        for disease, count in sorted_diseases[:20]:
            print(f"  {disease}: {count} 个样本")
        
        # 统计分布
        single_sample = sum(1 for _, count in disease_counts.items() if count == 1)
        few_samples = sum(1 for _, count in disease_counts.items() if 2 <= count <= 5)
        many_samples = sum(1 for _, count in disease_counts.items() if count > 5)
        
        print(f"\n样本分布:")
        print(f"  只有1个样本的疾病: {single_sample}")
        print(f"  有2-5个样本的疾病: {few_samples}")
        print(f"  有5个以上样本的疾病: {many_samples}")


# 使用示例
if __name__ == "__main__":
    # 初始化生成器
    generator = DiseaseDescriptionPairGenerator(
        "/Users/frozen/PycharmProjects/Medical_Image_Classification/Medical-Dialogue-Dataset-Chinese"
    )
    
    # 加载数据（可以先只加载部分文件测试）
    records = generator.load_all_files(max_files=30)  # 只加载前3个文件测试
    
    # 分析疾病分布
    generator.analyze_disease_distribution()
    
    # 创建训练配对（使用多种策略）
    pairs = generator.create_disease_description_pairs(
        output_file="disease_description_pairs.json",
        strategies=['simple', 'with_duration', 'multi_field', 'augmented']
    )
    
    # 显示几个样本
    print("\n" + "="*50)
    print("样本示例")
    print("="*50)
    
    for i, pair in enumerate(pairs[:5]):
        print(f"\n样本 {i+1}:")
        print(f"策略: {pair['strategy']}")
        print(f"类型: {pair['type']}")
        print(f"标签: {pair['label']}")
        print(f"Text1 (病情描述): {pair['text1'][:100]}...")
        print(f"Text2 (疾病名称): {pair['text2']}")