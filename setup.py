"""
Setup script for Neuroplastic Transformer Benchmark Suite
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neuroplastic-transformer-benchmark",
    version="1.0.0",
    author="Neuroplastic Transformer Team",
    author_email="neuroplastic-transformer@example.com",
    description="Comprehensive benchmark suite for Neuroplastic Transformer biologically-inspired computing systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuroplastic-transformer/benchmarking",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "performance": [
            "torch-audio>=2.0.0",
            "torch-vision>=0.15.0",
            "apex>=0.1.0",
            "flash-attn>=2.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuroplastic-benchmark=benchmarking_framework.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.json", "prompts/*.txt", "utils/*.py"],
    },
    keywords="ai, benchmarking, transformers, biologically-inspired computing, neuroplastic, performance",
    project_urls={
        "Bug Reports": "https://github.com/neuroplastic-transformer/benchmarking/issues",
        "Source": "https://github.com/neuroplastic-transformer/benchmarking",
        "Documentation": "https://neuroplastic-transformer.readthedocs.io",
    },
)