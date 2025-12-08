#!/usr/bin/env python3
"""
多模态医学图像分类主训练脚本入口
"""

import sys
from pathlib import Path

# 添加MIC包到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入MIC包中的主训练模块
from MIC.src.main import main

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)