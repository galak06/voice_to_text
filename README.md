# 🎤 Hebrew Voice to Text - Final Version

**Clean, streamlined Hebrew audio transcription with speaker separation and Word document output.**

## ✨ Features

- 🗣️ **Hebrew transcription** using OpenAI Whisper
- 👥 **Speaker separation** by paragraphs (alternating speakers)
- 📄 **Word document output** (.docx) with professional formatting
- 🎯 **Multiple model sizes** (tiny, base, small, medium, large)
- 📊 **Progress bars** for model loading and transcription
- 🎨 **Rich console output** with detailed information
- 🔧 **Easy to use** with simple command line interface

## 📁 Project Structure

```
voice_to_text/
├── transcribe_hebrew.py          # Main transcription script
├── run_hebrew.sh                 # Runner script
├── create_test_audio.py          # Test audio generator
├── setup.py                      # Package setup
├── install.sh                    # Environment installer
├── environment.yml               # Conda environment
├── requirements.txt              # Python dependencies
├── .env                          # Hugging Face token (create this)
├── voice/                        # Input audio files
│   ├── rachel_1.wav             # Original Hebrew audio
│   └── test_conversation.wav    # Test audio (30s, 2 voices)
└── output/                       # Generated files (with timestamps)
    ├── rachel_1_hebrew_speakers_tiny_20250730_194340.docx
    └── rachel_1_hebrew_speakers_large_20250730_195000.docx
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate voice-to-text
```

### 2. Configure Hugging Face Token
```bash
# Create .env file with your token
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```

### 3. Run Transcription
```bash
# Fastest (for testing)
./run_hebrew.sh --model tiny

# Good balance (production)
./run_hebrew.sh --model base

# Best quality (slower)
./run_hebrew.sh --model medium

# With GPU acceleration (if available)
./run_hebrew.sh --model tiny --device cuda

# Parallel processing for multiple files (minimum 2 workers)
./run_hebrew.sh --model tiny --workers 4

# Performance optimized (default)
./run_hebrew.sh --model tiny --process-folder

# Minimal memory usage
./run_hebrew.sh --model tiny --no-cache --no-optimize

# Raw text without speaker separation
./run_hebrew.sh --model tiny --no-text-improvements
```

## 📋 Usage Examples

### Basic Usage
```bash
# Process all audio files in voice/ folder (with timestamps)
./run_hebrew.sh

# Use specific model
./run_hebrew.sh --model medium

# Process specific file (with timestamp)
./run_hebrew.sh voice/test_conversation.wav

# Process specific file with custom output
./run_hebrew.sh voice/test_conversation.wav output/test_output.docx
```

### Model Options
- **tiny**: ~39MB, fastest, basic quality
- **base**: ~74MB, fast, good quality  
- **small**: ~244MB, balanced
- **medium**: ~769MB, good quality (default)
- **large**: ~1550MB, best quality, slowest

### Speed Optimizations
- **GPU acceleration**: 10x faster with CUDA
- **FP16 optimization**: 2x faster on GPU
- **Model caching**: Faster reloads (~/.cache/whisper)
- **Performance optimizations**: PyTorch optimizations, thread tuning
- **Parallel processing**: Minimum 2 workers (default: 2, recommended: 4)
- **Fast settings**: Optimized for tiny model
- **Memory optimization**: Disable cache/optimizations with `--no-cache --no-optimize`
- **Text improvements**: Speaker separation and paragraph formatting (disable with `--no-text-improvements`)

## 🎯 Output Format

### Word Document Structure
```
Hebrew Voice Transcription
Audio File: rachel_1.wav
Model: medium | Language: Hebrew | Speakers: 2
____________________________________________________________

Speaker 1
Sentence 1: [Hebrew text in RTL]

Speaker 2
Sentence 2: [Hebrew text in RTL]

Speaker 1
Sentence 3: [Hebrew text in RTL]

Speaker 2
Sentence 4: [Hebrew text in RTL]
...

____________________________________________________________
Summary: X sentences from Speaker 1, Y sentences from Speaker 2
```

### Speaker Separation Logic
- Text is split into individual sentences
- Each sentence alternates between speakers: Speaker 1 → Speaker 2 → Speaker 1 → etc.
- Speakers alternate in chronological order
- Output maintains time order: Speaker 1 sentence, then Speaker 2 sentence, then Speaker 1 sentence, etc.

## 🔧 Technical Details

### Dependencies
- **Python 3.10+**
- **OpenAI Whisper** (transcription)
- **python-docx** (Word document generation)
- **rich** (console output)
- **python-dotenv** (environment variables)
- **scipy** (test audio generation)

### Configuration
- **Language**: Hebrew (he)
- **Temperature**: 0.0 (most accurate)
- **Best of**: 5 candidates (1 for tiny model)
- **Beam size**: 5 (1 for tiny model)
- **Word timestamps**: Disabled
- **Initial prompt**: "This is a Hebrew conversation between two speakers."
- **FP16**: Enabled on GPU for 2x speed
- **Device**: Auto-detect GPU/CPU

## 🧪 Testing

### Create Test Audio
```bash
python create_test_audio.py
```
Creates `voice/test_conversation.wav` (30 seconds, 2 alternating voices)

### Test Transcription
```bash
# Quick test with tiny model
./run_hebrew.sh --model tiny

# Test with GPU (if available)
./run_hebrew.sh --model tiny --device cuda

# Test parallel processing
./run_hebrew.sh --model tiny --workers 2
```

## 📊 Quality Metrics

The system provides quality scoring based on:
- Text length
- Hebrew character density
- Punctuation and structure
- Overall coherence

Quality levels:
- **8.0-10.0**: 🎯 Excellent quality
- **6.0-7.9**: ✅ Good quality  
- **0.0-5.9**: ⚠️ Quality could be improved

## 🎨 Features

### Progress Tracking
- Model loading progress bar
- Transcription progress with time estimation
- Rich console tables and status indicators

### Professional Output
- Word document with proper Hebrew RTL formatting
- Speaker sections with color-coded paragraph numbers
- Document metadata and summary
- Clean, readable formatting

### Error Handling
- Graceful handling of missing files
- Clear error messages
- Environment validation
- Model access verification

## 🔄 Workflow

1. **Place audio files** in `voice/` folder
2. **Run transcription** with desired model size
3. **Get Word document** in `output/` folder
4. **Review speaker separation** and formatting
5. **Adjust model size** if needed for quality/speed

## 📝 Notes

- **GPU acceleration** available if CUDA is installed
- **Hugging Face token** required for some advanced features
- **Hebrew RTL** text properly formatted in Word documents
- **Speaker separation** is heuristic-based (not true diarization)
- **File formats** supported: .wav, .mp3, .m4a, .flac, .aac

## 🎯 Success Metrics

✅ **Clean, streamlined codebase**  
✅ **Professional Word document output**  
✅ **Speaker separation by paragraphs**  
✅ **Multiple model size options**  
✅ **Speed optimizations (GPU, FP16, parallel)**  
✅ **Progress tracking and rich UI**  
✅ **Test audio generation**  
✅ **Comprehensive documentation**  

---

**Final Version**: Clean, efficient, and ready for production use! 🚀 