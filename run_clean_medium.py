#!/usr/bin/env python3
"""
Clean output folders and run medium model transcription
"""
import subprocess
import sys
from pathlib import Path

def clean_output_folders():
    """Clean all output folders before running transcription"""
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

def run_medium_transcription():
    """Run the medium model transcription"""
    print("🎵 Starting medium model transcription...")
    
    try:
        result = subprocess.run([
            sys.executable, "process_chunks.py", 
            "voice_chunks", "--model", "medium"
        ], check=True)
        print("✅ Transcription completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Transcription failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Transcription cancelled by user")
        return False

if __name__ == "__main__":
    print("🚀 Starting clean transcription process...")
    
    # Step 1: Clean output folders
    clean_output_folders()
    
    # Step 2: Run transcription
    success = run_medium_transcription()
    
    if success:
        print("🎯 Process completed successfully!")
    else:
        print("💥 Process failed or was interrupted")
        sys.exit(1) 