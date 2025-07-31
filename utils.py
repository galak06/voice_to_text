#!/usr/bin/env python3
"""
Utility functions for the voice-to-text project.
"""

import re
from pathlib import Path
from typing import List


def natural_sort_key(text: str) -> List:
    """
    Generate a key for natural sorting that handles numbers correctly.
    
    Args:
        text: String to generate sort key for
        
    Returns:
        List of strings and integers for natural sorting
    """
    def atoi(text):
        return int(text) if text.isdigit() else text
    
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def natural_sort_files(files: List[Path]) -> List[Path]:
    """
    Sort files using natural sorting (handles numbers correctly).
    
    Args:
        files: List of file paths to sort
        
    Returns:
        Sorted list of file paths
    """
    return sorted(files, key=lambda x: natural_sort_key(x.name))


def validate_directory(path: Path, check_writable: bool = False) -> bool:
    """
    Validate that a path is a directory and optionally writable.
    
    Args:
        path: Path to validate
        check_writable: Whether to check if directory is writable
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not path.exists():
            return False
        if not path.is_dir():
            return False
        if check_writable:
            # Try to create a test file
            test_file = path / ".test_write_permission"
            test_file.touch()
            test_file.unlink()
        return True
    except (OSError, PermissionError):
        return False


# Import required modules at module level
import gc
import torch

def cleanup_resources():
    """
    Clean up system resources (memory, GPU cache, etc.).
    """
    # Force garbage collection
    gc.collect()
    
    # Clear PyTorch cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Reset thread count to default
    torch.set_num_threads(torch.get_num_threads())


def safe_file_operation(func):
    """
    Decorator to safely handle file operations with proper cleanup.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling and cleanup
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\n⚠️  Operation cancelled by user")
            cleanup_resources()
            raise
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            cleanup_resources()
            raise
        except PermissionError as e:
            print(f"❌ Permission denied: {e}")
            cleanup_resources()
            raise
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            cleanup_resources()
            raise
    
    return wrapper 