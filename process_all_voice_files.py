#!/usr/bin/env python3
"""
Process all voice files from the voice folder one by one
Splits each file into chunks and transcribes them
"""
import subprocess
import sys
from pathlib import Path
import shutil

def clean_output_folders():
    """Clean all output folders before processing"""
    print("🧹 Cleaning output folders...")
    
    output_dirs = [
        "output/chunks",
        "output/merged", 
        "output/text",
        "output/word",
        "output/temp"
    ]
    
    for dir_path in output_dirs:
        path = Path(dir_path)
        if path.exists():
            # Remove all files in directory
            for file_path in path.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    print(f"  Deleted: {file_path}")
    
    print("✅ Output folders cleaned")

def get_voice_files():
    """Get all voice files from the voice folder"""
    voice_dir = Path("voice")
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.aac'}
    
    voice_files = [
        f for f in voice_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]
    
    return sorted(voice_files)

def split_audio_file(audio_file):
    """Split a single audio file into chunks"""
    print(f"🎵 Splitting audio file: {audio_file}")
    
    try:
        result = subprocess.run([
            sys.executable, "split_audio.py", str(audio_file), "--output-dir", "output/chunks"
        ], check=True, capture_output=True, text=True)
        
        print(f"✅ Successfully split {audio_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error splitting {audio_file}: {e}")
        return False

def transcribe_chunks(model="medium"):
    """Transcribe all chunks with specified model"""
    print(f"🎤 Transcribing chunks with {model} model...")
    
    try:
        result = subprocess.run([
            sys.executable, "process_chunks.py", "output/chunks", "--model", model
        ], check=True)
        
        print(f"✅ Successfully transcribed chunks with {model} model")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error transcribing chunks: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Transcription cancelled by user")
        return False

def merge_results():
    """Merge all transcription results"""
    print("🔗 Merging transcription results...")
    
    try:
        result = subprocess.run([
            sys.executable, "merge_only.py", "output/chunks"
        ], check=True)
        
        print("✅ Successfully merged results")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error merging results: {e}")
        return False

def process_single_voice_file(audio_file, model="medium"):
    """Process a single voice file completely"""
    print(f"\n{'='*60}")
    print(f"🎯 Processing: {audio_file.name}")
    print(f"{'='*60}")
    
    # Step 1: Split audio into chunks (don't clean output folders)
    if not split_audio_file(audio_file):
        print(f"❌ Failed to split {audio_file.name}")
        return False
    
    # Step 2: Transcribe chunks
    if not transcribe_chunks(model):
        print(f"❌ Failed to transcribe chunks for {audio_file.name}")
        return False
    
    # Step 3: Merge results
    if not merge_results():
        print(f"❌ Failed to merge results for {audio_file.name}")
        return False
    
    print(f"✅ Successfully completed processing {audio_file.name}")
    return True

def main():
    """Main function to process all voice files"""
    print("🚀 Starting processing of all voice files...")
    
    # Get all voice files
    voice_files = get_voice_files()
    
    if not voice_files:
        print("❌ No voice files found in voice/ folder")
        return
    
    print(f"📁 Found {len(voice_files)} voice files:")
    for i, file in enumerate(voice_files, 1):
        print(f"  {i}. {file.name}")
    
    # Use medium model by default (no user input required)
    print("\n🤖 Using medium model (recommended for good quality)")
    
    model = "medium"
    print(f"🎯 Using {model} model")
    
    # Clean output folders once at the beginning
    print(f"\n🧹 Cleaning output folders once at the beginning...")
    clean_output_folders()
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, audio_file in enumerate(voice_files, 1):
        print(f"\n📊 Progress: {i}/{len(voice_files)}")
        
        if process_single_voice_file(audio_file, model):
            successful += 1
        else:
            failed += 1
        
        # Continue automatically to next file
        if i < len(voice_files):
            print(f"\n🔄 Continuing automatically to next file...")
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Total files: {len(voice_files)}")
    
    if successful > 0:
        print(f"\n🎯 Processing complete! Check output/merged/ for final results")
    else:
        print(f"\n💥 No files were processed successfully")

if __name__ == "__main__":
    main() 