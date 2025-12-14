import json
import os
from pathlib import Path

def process_medical_dialogue_files(input_dir, output_file):
    """
    处理Medical-Dialogue-Dataset-Chinese文件夹中的所有txt文件
    将每个id的内容作为一个文档，组织成JSON格式
    
    Args:
        input_dir: 输入文件夹路径
        output_file: 输出JSON文件路径
    """
    # 需要处理的文件列表
    txt_files = [
        '2010.txt', '2012.txt', '2013.txt', '2015.txt', 
        '2016.txt', '2018.txt', '2019.txt', '2020.txt'
    ]
    
    all_documents = []
    
    for txt_file in txt_files:
        file_path = os.path.join(input_dir, txt_file)
        if not os.path.exists(file_path):
            print(f"文件不存在，跳过: {file_path}")
            continue
            
        print(f"正在处理: {txt_file}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            current_document = []
            document_count = 0
            
            for line in f:
                line = line.rstrip('\n')
                
                # 检查是否是新的id开始（但不是第一行）
                if line.startswith('id=') and current_document:
                    # 保存之前的文档
                    document_text = '\n'.join(current_document)
                    all_documents.append({"text": document_text})
                    document_count += 1
                    
                    # 开始新文档
                    current_document = [line]
                else:
                    current_document.append(line)
            
            # 保存最后一个文档
            if current_document:
                document_text = '\n'.join(current_document)
                all_documents.append({"text": document_text})
                document_count += 1
        
        print(f"  完成，提取了 {document_count} 个文档")
    
    # 写入JSON文件
    print(f"\n正在写入JSON文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)
    
    print(f"完成！总共处理了 {len(all_documents)} 个文档")
    print(f"输出文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # 设置路径
    input_directory = "/Users/frozen/PycharmProjects/Medical_Image_Classification/Medical-Dialogue-Dataset-Chinese"
    output_json_file = "/Users/frozen/PycharmProjects/Medical_Image_Classification/Medical-Dialogue-Dataset-Chinese/medical_pretrain_formatted.json"
    
    # 处理文件
    process_medical_dialogue_files(input_directory, output_json_file)