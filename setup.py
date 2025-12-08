"""
MIC (Medical Image Classification) Package Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README文件
current_dir = Path(__file__).parent
long_description = (current_dir / "README.md").read_text(encoding="utf-8")

# 读取requirements文件
requirements = []
requirements_file = current_dir / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="medical-info-classification",
    version="1.0.0",
    author="frozenleaves",
    author_email="example@example.com",
    description="多模态医学信息分类",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/frozenleaves/Medical_Info_Classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "visualization": ["plotly", "dash"],
        "wsi": ["openslide-python"],
    },
    entry_points={
        "console_scripts": [
            "mic-train=MIC.src.main:main",
            "mic-inference=MIC.src.inference:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
