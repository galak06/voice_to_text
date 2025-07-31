#!/bin/bash
# Hebrew Voice to Text with Speaker Separation - Runner Script
# Processes voice files and outputs Hebrew RTL text to .docx files with clear speaker separation
PYTHON_PATH="/opt/homebrew/Caskroom/miniconda/base/envs/voice-to-text/bin/python"
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ Error: Python interpreter not found at $PYTHON_PATH"
    echo "   Make sure the conda environment is created: conda env create -f environment.yml"
    exit 1
fi
echo "🎤 Running Hebrew Voice to Text with Speaker Separation (Word Document)..."
$PYTHON_PATH transcribe_hebrew.py "$@" 