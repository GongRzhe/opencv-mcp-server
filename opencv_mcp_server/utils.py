import cv2
import numpy as np
import os
import logging
import datetime
import subprocess
import platform
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger("opencv-mcp-server.utils")

# Utility Functions
def get_image_info(image: np.ndarray) -> Dict[str, Any]:
    """
    Get basic information about an image
    
    Args:
        image: OpenCV image
        
    Returns:
        Dict: Image information including dimensions, channels, etc.
    """
    if image is None:
        raise ValueError("Image is None")
    
    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    dtype = str(image.dtype)
    size_bytes = image.nbytes
    
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": dtype,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2)
    }

def get_timestamp() -> str:
    """
    Get current timestamp as a string
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def open_image_with_system_viewer(image_path: str) -> None:
    """
    Open an image with the system's default image viewer
    
    Args:
        image_path: Path to the image file
    """
    # Platform-specific image opening commands
    system = platform.system()
    
    try:
        if system == 'Windows':
            os.startfile(image_path)
        elif system == 'Darwin':  # macOS
            subprocess.call(['open', image_path])
        else:  # Linux and other Unix-like systems
            subprocess.call(['xdg-open', image_path])
        
        logger.info(f"Opened image: {image_path}")
    except Exception as e:
        logger.error(f"Error opening image with system viewer: {e}")
        # Continue execution even if display fails

def open_video_with_system_viewer(video_path: str) -> None:
    """
    Open a video with the system's default video player
    
    Args:
        video_path: Path to the video file
    """
    # Platform-specific video opening commands
    system = platform.system()
    
    try:
        if system == 'Windows':
            os.startfile(video_path)
        elif system == 'Darwin':  # macOS
            subprocess.call(['open', video_path])
        else:  # Linux and other Unix-like systems
            subprocess.call(['xdg-open', video_path])
        
        logger.info(f"Opened video: {video_path}")
    except Exception as e:
        logger.error(f"Error opening video with system viewer: {e}")
        # Continue execution even if display fails

def get_video_output_folder(video_path: str, operation: str) -> str:
    """
    Create and return a folder for storing video processing outputs
    
    Args:
        video_path: Path to the video file
        operation: Name of operation being performed
        
    Returns:
        str: Path to the output folder
    """
    # Get directory of original video
    directory = os.path.dirname(video_path) or '.'
    
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create folder name with video name, operation and timestamp
    timestamp = get_timestamp()
    folder_name = f"{video_name}_{operation}_{timestamp}"
    folder_path = os.path.join(directory, folder_name)
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    return folder_path

def save_and_display(img: np.ndarray, original_path: str, operation: str) -> str:
    """
    Save image to file and display it using system's default image viewer
    
    Args:
        img: OpenCV image
        original_path: Path to original image
        operation: Name of operation performed
        
    Returns:
        str: Path to saved image
    """
    # Determine if this is a video frame by checking if the path contains specific markers
    is_video_frame = any(marker in os.path.basename(original_path) for marker in 
                         ["_frame_", "_track_", "_motion_"])
    
    # Get filename without extension
    base_name = os.path.basename(original_path)
    name_parts = os.path.splitext(base_name)
    
    # Create new filename with operation and timestamp
    timestamp = get_timestamp()
    new_filename = f"{name_parts[0]}_{operation}_{timestamp}{name_parts[1]}"
    
    # Get directory based on whether it's a video frame or regular image
    if is_video_frame:
        # Use the same directory as the original
        directory = os.path.dirname(original_path)
    else:
        # Get directory of original image
        directory = os.path.dirname(original_path) or '.'
    
    new_path = os.path.join(directory, new_filename)
    
    # Save image
    cv2.imwrite(new_path, img)
    
    # Display image using system's default image viewer
    open_image_with_system_viewer(new_path)
    
    return new_path
