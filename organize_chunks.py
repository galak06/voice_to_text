#!/usr/bin/env python3
"""
Organize chunk files into individual folders by filename.
Each chunk gets its own folder containing both audio and transcription files.
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def organize_chunks(voice_chunks_dir: Path, output_chunks_dir: Path, organized_dir: Path = Path("output/organized")):
    """
    Organize chunk files into individual folders by filename.
    
    Args:
        voice_chunks_dir: Directory containing audio chunk files
        output_chunks_dir: Directory containing transcription chunk files
        organized_dir: Directory to create organized structure
    """
    try:
        # Create organized directory
        organized_dir.mkdir(exist_ok=True)
        
        # Find all audio chunks
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
        audio_files = [f for f in voice_chunks_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in audio_extensions]
        
        # Find all transcription files
        transcription_files = list(output_chunks_dir.glob("*.docx"))
        transcription_files = [f for f in transcription_files if not f.name.startswith("~$")]
        
        print(f"📁 Found {len(audio_files)} audio files and {len(transcription_files)} transcription files")
        
        # Create a mapping of chunk names to their files
        chunk_mapping = {}
        
        # Process audio files
        for audio_file in audio_files:
            # Extract chunk name (e.g., "rachel_1_chunk_001" from "rachel_1_chunk_001.wav")
            chunk_name = audio_file.stem
            if chunk_name not in chunk_mapping:
                chunk_mapping[chunk_name] = {"audio": None, "transcription": None, "timestamp": None}
            chunk_mapping[chunk_name]["audio"] = audio_file
        
        # Process transcription files
        for trans_file in transcription_files:
            # Extract chunk name and timestamp from transcription filename
            # e.g., "rachel_1_chunk_001_transcribed_20250730_214558" -> "rachel_1_chunk_001" and "20250730_214558"
            parts = trans_file.stem.split("_transcribed_")
            if len(parts) >= 2:
                chunk_name = parts[0]
                timestamp = parts[1]
                if chunk_name not in chunk_mapping:
                    chunk_mapping[chunk_name] = {"audio": None, "transcription": None, "timestamp": None}
                chunk_mapping[chunk_name]["transcription"] = trans_file
                chunk_mapping[chunk_name]["timestamp"] = timestamp
            elif len(parts) >= 1:
                chunk_name = parts[0]
                if chunk_name not in chunk_mapping:
                    chunk_mapping[chunk_name] = {"audio": None, "transcription": None, "timestamp": None}
                chunk_mapping[chunk_name]["transcription"] = trans_file
        
        # Create folders and copy files
        organized_count = 0
        for chunk_name, files in chunk_mapping.items():
            # Create chunk folder with timestamp if available
            if files["timestamp"]:
                folder_name = f"{chunk_name}_{files['timestamp']}"
            else:
                folder_name = chunk_name
            
            chunk_folder = organized_dir / folder_name
            chunk_folder.mkdir(exist_ok=True)
            
            print(f"📂 Creating folder: {folder_name}")
            
            # Copy audio file if exists
            if files["audio"]:
                audio_dest = chunk_folder / files["audio"].name
                shutil.copy2(files["audio"], audio_dest)
                print(f"  🎵 Copied audio: {files['audio'].name}")
            
            # Copy transcription file if exists
            if files["transcription"]:
                trans_dest = chunk_folder / files["transcription"].name
                shutil.copy2(files["transcription"], trans_dest)
                print(f"  📄 Copied transcription: {files['transcription'].name}")
            
            # Create a summary file for the chunk
            summary_file = chunk_folder / "chunk_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Chunk: {chunk_name}\n")
                if files["timestamp"]:
                    f.write(f"Timestamp: {files['timestamp']}\n")
                f.write("=" * 50 + "\n")
                if files["audio"]:
                    f.write(f"Audio file: {files['audio'].name}\n")
                    f.write(f"Audio size: {files['audio'].stat().st_size / 1024 / 1024:.2f} MB\n")
                if files["transcription"]:
                    f.write(f"Transcription file: {files['transcription'].name}\n")
                    f.write(f"Transcription size: {files['transcription'].stat().st_size / 1024:.2f} KB\n")
                f.write(f"Created: {chunk_folder}\n")
            
            print(f"  📝 Created summary: chunk_summary.txt")
            organized_count += 1
        
        # Create overall summary
        overall_summary = organized_dir / "organization_summary.txt"
        with open(overall_summary, 'w', encoding='utf-8') as f:
            f.write("Chunk Organization Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total chunks organized: {organized_count}\n")
            f.write(f"Source audio directory: {voice_chunks_dir}\n")
            f.write(f"Source transcription directory: {output_chunks_dir}\n")
            f.write(f"Organized directory: {organized_dir}\n")
            f.write("\nChunk folders created:\n")
            for chunk_name, files in chunk_mapping.items():
                if files["timestamp"]:
                    folder_name = f"{chunk_name}_{files['timestamp']}"
                else:
                    folder_name = chunk_name
                f.write(f"  - {folder_name}/\n")
        
        print(f"\n✅ Organization complete!")
        print(f"📁 Organized {organized_count} chunks in: {organized_dir}")
        print(f"📄 Overall summary: {overall_summary}")
        
        return organized_count
        
    except Exception as e:
        print(f"❌ Error organizing chunks: {e}")
        return 0


def main():
    """Main function to handle command line arguments and run organization."""
    parser = argparse.ArgumentParser(
        description="Organize chunk files into individual folders by filename",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s voice_chunks output_chunks
  %(prog)s voice_chunks output_chunks --output my_organized_chunks
        """
    )
    
    parser.add_argument("voice_chunks_dir", help="Directory containing audio chunk files")
    parser.add_argument("output_chunks_dir", help="Directory containing transcription chunk files")
    parser.add_argument("--output", default="organized_chunks",
                       help="Output directory for organized structure (default: organized_chunks)")
    
    args = parser.parse_args()
    
    # Import validation utility
    from utils import validate_directory
    
    # Validate directories
    voice_chunks_path = Path(args.voice_chunks_dir)
    output_chunks_path = Path(args.output_chunks_dir)
    
    if not validate_directory(voice_chunks_path):
        print(f"❌ Invalid voice chunks directory: {args.voice_chunks_dir}")
        print(f"   Directory must exist and be accessible")
        return
    
    if not validate_directory(output_chunks_path):
        print(f"❌ Invalid output chunks directory: {args.output_chunks_dir}")
        print(f"   Directory must exist and be accessible")
        return
    
    print(f"🗂️  Organizing chunk files...")
    organize_chunks(Path(args.voice_chunks_dir), Path(args.output_chunks_dir), Path(args.output))
    print(f"🎯 Organization complete!")


if __name__ == "__main__":
    main() 