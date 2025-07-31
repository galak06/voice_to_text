#!/usr/bin/env python3
"""
Setup script for Voice to Text project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="voice-to-text",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Hebrew audio transcription with speaker diarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/voice-to-text",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "voice-to-text=transcribe:main",
        ],
    },
) 