#!/bin/bash

# Voice to Text - Installation Script
# This script sets up the conda environment and installs all dependencies

set -e  # Exit on any error

echo "🎤 Voice to Text - Installation Script"
echo "======================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "voice-to-text"; then
    echo "⚠️  Environment 'voice-to-text' already exists."
    read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n voice-to-text
    else
        echo "📝 Using existing environment. Activate it with: conda activate voice-to-text"
        exit 0
    fi
fi

echo "🔧 Creating conda environment..."
conda env create -f environment.yml

echo "✅ Environment created successfully!"
echo ""
echo "🚀 Next steps:"
echo "1. Activate the environment:"
echo "   conda activate voice-to-text"
echo ""
echo "2. Set up Hugging Face authentication:"
echo "   python setup_huggingface.py"
echo ""
echo "3. Test the installation:"
echo "   ./run.sh --help"
echo ""
echo "4. Try transcribing the example file:"
echo "   ./run.sh --input voice/rachel_1.wav --output output/rachel_1_transcript.txt"
echo ""
echo "📚 For more information, see README.md" 