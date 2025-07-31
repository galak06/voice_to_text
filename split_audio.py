#!/usr/bin/env python3
"""
Audio File Splitter for Large Hebrew Audio Files
Splits large audio files into smaller chunks for faster transcription processing
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

import librosa
import soundfile as sf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Rich console for better output
console = Console()


class AudioSplitter:
    """
    Splits large audio files into smaller chunks for faster processing.
    """
    
    def __init__(self, chunk_duration: int = 300, overlap: int = 30):
        """
        Initialize the audio splitter.
        
        Args:
            chunk_duration: Duration of each chunk in seconds (default: 5 minutes)
            overlap: Overlap between chunks in seconds (default: 30 seconds)
        """
        # Validate parameters
        if overlap >= chunk_duration:
            raise ValueError(f"Overlap ({overlap}s) must be less than chunk duration ({chunk_duration}s)")
        if chunk_duration <= 0 or overlap < 0:
            raise ValueError("Chunk duration and overlap must be positive")
        
        self.chunk_duration = chunk_duration
        self.overlap = overlap
    
    def split_audio_file(self, input_path: str, output_dir: str = "output/chunks") -> List[str]:
        """
        Split a large audio file into smaller chunks.
        
        Args:
            input_path: Path to the input audio file
            output_dir: Directory to save the chunks
            
        Returns:
            List of paths to the created chunk files
        """
        try:
            input_path = Path(input_path)
            output_dir = Path(output_dir)
            
            # Create output directory
            output_dir.mkdir(exist_ok=True)
            
            console.print(f"[blue]Loading audio file: {input_path}[/blue]")
            
            # Load audio file
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Loading audio file...", total=100)
                
                # Load audio with librosa
                y, sr = librosa.load(str(input_path), sr=None)
                
                progress.update(task, completed=50)
                
                # Calculate duration
                duration = len(y) / sr
                console.print(f"[green]Audio duration: {duration:.1f} seconds ({duration/60:.1f} minutes)[/green]")
                
                # Calculate number of chunks
                chunk_samples = int(self.chunk_duration * sr)
                overlap_samples = int(self.overlap * sr)
                step_samples = chunk_samples - overlap_samples
                
                num_chunks = max(1, int((len(y) - overlap_samples) / step_samples) + 1)
                
                progress.update(task, completed=100)
            
            console.print(f"[blue]Splitting into {num_chunks} chunks...[/blue]")
            
            chunk_files = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Creating chunks...", total=num_chunks)
                
                for i in range(num_chunks):
                    start_sample = i * step_samples
                    end_sample = min(start_sample + chunk_samples, len(y))
                    
                    # Extract chunk
                    chunk_audio = y[start_sample:end_sample]
                    
                    # Create output filename
                    chunk_filename = f"{input_path.stem}_chunk_{i+1:03d}.wav"
                    chunk_path = output_dir / chunk_filename
                    
                    # Save chunk
                    sf.write(str(chunk_path), chunk_audio, sr)
                    
                    chunk_files.append(str(chunk_path))
                    progress.update(task, advance=1)
                    
                    console.print(f"[green]Created: {chunk_filename} ({len(chunk_audio)/sr:.1f}s)[/green]")
            
            # Create a summary file
            summary_path = output_dir / "chunks_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Original file: {input_path}\n")
                f.write(f"Original duration: {duration:.1f} seconds\n")
                f.write(f"Number of chunks: {num_chunks}\n")
                f.write(f"Chunk duration: {self.chunk_duration} seconds\n")
                f.write(f"Overlap: {self.overlap} seconds\n\n")
                f.write("Chunk files:\n")
                for i, chunk_file in enumerate(chunk_files, 1):
                    f.write(f"{i}. {Path(chunk_file).name}\n")
            
            console.print(f"[green]✅ Successfully split audio into {num_chunks} chunks[/green]")
            console.print(f"[blue]Chunks saved in: {output_dir}[/blue]")
            console.print(f"[blue]Summary file: {summary_path}[/blue]")
            
            return chunk_files
            
        except Exception as e:
            console.print(f"[red]❌ Error splitting audio: {e}[/red]")
            raise
    
    def show_split_info(self, input_path: str):
        """Show information about the splitting process."""
        try:
            input_path = Path(input_path)
            
            # Load audio to get duration
            y, sr = librosa.load(str(input_path), sr=None)
            duration = len(y) / sr
            
            # Calculate chunks
            chunk_samples = int(self.chunk_duration * sr)
            overlap_samples = int(self.overlap * sr)
            step_samples = chunk_samples - overlap_samples
            num_chunks = max(1, int((len(y) - overlap_samples) / step_samples) + 1)
            
            table = Table(title="Audio Splitting Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Description", style="yellow")
            
            table.add_row("Input File", input_path.name, "Original audio file")
            table.add_row("File Size", f"{input_path.stat().st_size / (1024*1024):.1f} MB", "File size in megabytes")
            table.add_row("Duration", f"{duration:.1f} seconds", f"{duration/60:.1f} minutes")
            table.add_row("Sample Rate", f"{sr} Hz", "Audio sample rate")
            table.add_row("Chunk Duration", f"{self.chunk_duration} seconds", f"{self.chunk_duration/60:.1f} minutes")
            table.add_row("Overlap", f"{self.overlap} seconds", "Overlap between chunks")
            table.add_row("Number of Chunks", str(num_chunks), "Estimated chunks")
            table.add_row("Processing Time", f"{num_chunks * 2:.1f} minutes", "Estimated total processing time")
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]❌ Error analyzing audio: {e}[/red]")


def main():
    """Main function to handle command line arguments and run audio splitting."""
    parser = argparse.ArgumentParser(
        description="Split large audio files into smaller chunks for faster transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s voice/rachel_1.wav
  %(prog)s voice/rachel_1.wav --chunk-duration 180 --overlap 15
  %(prog)s voice/rachel_1.wav --info
        """
    )
    
    parser.add_argument("input_file", help="Input audio file path")
    parser.add_argument("--chunk-duration", type=int, default=300, 
                       help="Duration of each chunk in seconds (default: 300 = 5 minutes)")
    parser.add_argument("--overlap", type=int, default=30, 
                       help="Overlap between chunks in seconds (default: 30)")
    parser.add_argument("--output-dir", default="output/chunks", 
                       help="Output directory for chunks (default: output/chunks)")
    parser.add_argument("--info", action="store_true", 
                       help="Show splitting information without creating chunks")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        console.print(f"[red]❌ Input file not found: {args.input_file}[/red]")
        sys.exit(1)
    
    # Create splitter
    splitter = AudioSplitter(
        chunk_duration=args.chunk_duration,
        overlap=args.overlap
    )
    
    if args.info:
        # Show information only
        splitter.show_split_info(args.input_file)
    else:
        # Split the audio file
        console.print(f"🎵 Splitting audio file: {args.input_file}")
        chunk_files = splitter.split_audio_file(args.input_file, args.output_dir)
        
        console.print(f"\n[green]🎯 Next steps:[/green]")
        console.print(f"1. Process chunks with: ./run_hebrew.sh --model medium --workers 5 --input-file voice_chunks/chunk_001.wav")
        console.print(f"2. Or process all chunks in parallel")
        console.print(f"3. Merge results using the summary file")


if __name__ == "__main__":
    main() 