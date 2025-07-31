#!/usr/bin/env python3
"""
Checkpoint Manager for Voice-to-Text Processing
Manages progress persistence and resume functionality for long-running operations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoints for long-running transcription operations.
    """
    
    def __init__(self, checkpoint_file: str = "transcription_checkpoint.json"):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_file: Path to checkpoint file
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_data = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load existing checkpoint data."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        
        return {
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "status": "initialized",
            "total_chunks": 0,
            "processed_chunks": [],
            "failed_chunks": [],
            "current_chunk": None,
            "output_files": [],
            "settings": {}
        }
    
    def _save_checkpoint(self):
        """Save current checkpoint data."""
        try:
            self.checkpoint_data["last_updated"] = datetime.now().isoformat()
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Checkpoint saved to {self.checkpoint_file}")
        except IOError as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def initialize_session(self, total_chunks: int, settings: Dict[str, Any]):
        """
        Initialize a new processing session.
        
        Args:
            total_chunks: Total number of chunks to process
            settings: Processing settings
        """
        self.checkpoint_data.update({
            "status": "running",
            "total_chunks": total_chunks,
            "processed_chunks": [],
            "failed_chunks": [],
            "current_chunk": None,
            "output_files": [],
            "settings": settings
        })
        self._save_checkpoint()
    
    def start_chunk(self, chunk_name: str):
        """
        Mark a chunk as being processed.
        
        Args:
            chunk_name: Name of the chunk being processed
        """
        self.checkpoint_data["current_chunk"] = chunk_name
        self._save_checkpoint()
    
    def complete_chunk(self, chunk_name: str, output_file: str, status: str = "success"):
        """
        Mark a chunk as completed.
        
        Args:
            chunk_name: Name of the completed chunk
            output_file: Path to the output file
            status: Status of the chunk processing
        """
        if self.checkpoint_data["current_chunk"] == chunk_name:
            self.checkpoint_data["current_chunk"] = None
        
        chunk_info = {
            "name": chunk_name,
            "output_file": output_file,
            "status": status,
            "completed_at": datetime.now().isoformat()
        }
        
        if status == "success":
            self.checkpoint_data["processed_chunks"].append(chunk_info)
            self.checkpoint_data["output_files"].append(output_file)
        else:
            self.checkpoint_data["failed_chunks"].append(chunk_info)
        
        self._save_checkpoint()
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress information.
        
        Returns:
            Dictionary with progress information
        """
        total = self.checkpoint_data["total_chunks"]
        processed = len(self.checkpoint_data["processed_chunks"])
        failed = len(self.checkpoint_data["failed_chunks"])
        current = self.checkpoint_data["current_chunk"]
        
        return {
            "total_chunks": total,
            "processed_chunks": processed,
            "failed_chunks": failed,
            "current_chunk": current,
            "progress_percentage": (processed / total * 100) if total > 0 else 0,
            "remaining_chunks": total - processed - failed,
            "status": self.checkpoint_data["status"]
        }
    
    def can_resume(self) -> bool:
        """
        Check if processing can be resumed.
        
        Returns:
            True if resume is possible
        """
        return (
            self.checkpoint_data["status"] == "running" and
            self.checkpoint_data["total_chunks"] > 0 and
            len(self.checkpoint_data["processed_chunks"]) < self.checkpoint_data["total_chunks"]
        )
    
    def get_remaining_chunks(self, all_chunks: List[str]) -> List[str]:
        """
        Get list of chunks that still need processing.
        
        Args:
            all_chunks: List of all chunk names
            
        Returns:
            List of remaining chunk names
        """
        processed_names = {chunk["name"] for chunk in self.checkpoint_data["processed_chunks"]}
        failed_names = {chunk["name"] for chunk in self.checkpoint_data["failed_chunks"]}
        
        remaining = []
        for chunk in all_chunks:
            if chunk not in processed_names and chunk not in failed_names:
                remaining.append(chunk)
        
        return remaining
    
    def mark_completed(self):
        """Mark the entire session as completed."""
        self.checkpoint_data["status"] = "completed"
        self._save_checkpoint()
    
    def mark_failed(self, error: str):
        """
        Mark the session as failed.
        
        Args:
            error: Error description
        """
        self.checkpoint_data["status"] = "failed"
        self.checkpoint_data["error"] = error
        self._save_checkpoint()
    
    def cleanup(self):
        """Clean up checkpoint file."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.info(f"Checkpoint file cleaned up: {self.checkpoint_file}")
        except IOError as e:
            logger.warning(f"Failed to cleanup checkpoint: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the processing session.
        
        Returns:
            Dictionary with session summary
        """
        progress = self.get_progress()
        
        return {
            "session_info": {
                "created": self.checkpoint_data["created"],
                "last_updated": self.checkpoint_data["last_updated"],
                "status": self.checkpoint_data["status"]
            },
            "progress": progress,
            "settings": self.checkpoint_data["settings"],
            "output_files": self.checkpoint_data["output_files"],
            "processed_chunks": self.checkpoint_data["processed_chunks"],
            "failed_chunks": self.checkpoint_data["failed_chunks"]
        } 