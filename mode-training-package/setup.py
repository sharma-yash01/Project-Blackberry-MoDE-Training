#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for MoDE Budget Model package

Installation:
    pip install -e .
    pip install -e .[dev]  # With development dependencies
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="mode-budget-model",
    version="0.1.0",
    description="Mixture of Difficulty Experts (MoDE) Budget Model for Reasoning Economics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CSCI 566 Research Team",
    author_email="",
    url="https://github.com/yourusername/mode-budget-model",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "cloud": [
            "google-cloud-aiplatform>=1.38.0",
            "google-cloud-storage>=2.10.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "train-mode=scripts.train_mode:main",
            "train-supervised=scripts.train_supervised:main",
            "evaluate-mode=scripts.evaluate:main",
            "inference-mode=scripts.run_inference:main",
        ],
    },
)

