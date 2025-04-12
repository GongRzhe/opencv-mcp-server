"""
OpenCV MCP Server - Video Processing

This module provides video handling and analysis tools using OpenCV.
It includes functionality for extracting frames, detecting motion,
and tracking objects in videos.
"""

import cv2
import numpy as np
import os
import logging
import tempfile
from typing import Optional, List, Dict, Any, Tuple, Union

# Import utility functions from utils
from .utils import get_image_info, save_and_display, get_timestamp, get_video_output_folder, open_image_with_system_viewer

logger = logging.getLogger("opencv-mcp-server.video_processing")

# Video utility functions
def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get basic information about a video
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dict: Video information including dimensions, fps, frame count, etc.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Convert fourcc to readable format
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        
        # Calculate duration in seconds
        duration = frame_count / fps if fps > 0 else 0
        
        # Release the capture
        cap.release()
        
        return {
            "width": width,
            "height": height,
            "fps": float(fps),
            "frame_count": frame_count,
            "fourcc": fourcc,
            "duration_seconds": float(duration),
            "duration_formatted": f"{int(duration // 60):02d}:{int(duration % 60):02d}"
        }
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        raise ValueError(f"Failed to get video info: {str(e)}")

def detect_video_file(file_path: str) -> bool:
    """
    Detect if a file is a valid video file
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if file is a valid video, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            return False
            
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
            
        # Read the first frame
        ret, frame = cap.read()
        
        # Release the capture
        cap.release()
        
        return ret and frame is not None
    except Exception as e:
        logger.error(f"Error detecting video file: {str(e)}")
        return False

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

# Tool implementations
def extract_video_frames_tool(
    video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
    max_frames: int = 10,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to video file
        start_frame: Starting frame index (0-based)
        end_frame: Ending frame index (inclusive), if None extracts until the end
        step: Step size (extract every nth frame)
        max_frames: Maximum number of frames to extract
        output_dir: Directory to save extracted frames (if None, use auto-generated folder)
        
    Returns:
        Dict: Extracted frames information and paths
    """
    try:
        # Validate video file
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Get video info
        video_info = get_video_info(video_path)
        total_frames = video_info["frame_count"]
        
        # Validate and adjust parameters
        if start_frame < 0:
            start_frame = 0
        
        if end_frame is None or end_frame >= total_frames:
            end_frame = total_frames - 1
        
        if step < 1:
            step = 1
        
        # Set output directory
        if output_dir is None:
            # Create a dedicated folder for this extraction
            output_dir = get_video_output_folder(video_path, "frames")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculate number of frames to extract
        num_frames = min(
            (end_frame - start_frame) // step + 1,
            max_frames
        )
        
        # Extract frames
        frames = []
        
        # Set position to start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for i in range(num_frames):
            # Calculate the frame index
            frame_idx = start_frame + (i * step)
            
            # Skip if we've reached the end_frame
            if frame_idx > end_frame:
                break
                
            # Set position (to handle potential frame skipping issues)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read the frame
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame at index {frame_idx}")
                break
            
            # Get frame timestamp
            frame_timestamp = frame_idx / video_info["fps"]
            
            # Save frame to file
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_filename = f"{video_name}_frame_{frame_idx}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            
            # Add to frames list
            frames.append({
                "index": frame_idx,
                "timestamp_seconds": float(frame_timestamp),
                "timestamp_formatted": f"{int(frame_timestamp // 60):02d}:{int(frame_timestamp % 60):02d}.{int((frame_timestamp % 1) * 100):02d}",
                "path": frame_path
            })
        
        # Release the capture
        cap.release()
        
        return {
            "frames": frames,
            "frame_count": len(frames),
            "extraction_parameters": {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "step": step,
                "max_frames": max_frames
            },
            "video_info": video_info,
            "output_dir": output_dir
        }
        
    except Exception as e:
        logger.error(f"Error extracting video frames: {str(e)}")
        raise ValueError(f"Failed to extract video frames: {str(e)}")

def detect_motion_tool(
    frame1_path: str,
    frame2_path: str,
    threshold: int = 25,
    blur_size: int = 5,
    dilate_size: int = 5,
    min_area: int = 500,
    draw: bool = True,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> Dict[str, Any]:
    """
    Detect motion between two frames
    
    Args:
        frame1_path: Path to the first frame
        frame2_path: Path to the second frame
        threshold: Threshold for binary conversion
        blur_size: Kernel size for Gaussian blur
        dilate_size: Kernel size for dilation
        min_area: Minimum contour area to consider as motion
        draw: Whether to draw motion contours
        color: Color for drawing contours (BGR format)
        thickness: Thickness of contour lines
        
    Returns:
        Dict: Motion detection results and visualizations
    """
    try:
        # Read frames from paths
        frame1 = cv2.imread(frame1_path)
        if frame1 is None:
            raise ValueError(f"Failed to read image from path: {frame1_path}")
            
        frame2 = cv2.imread(frame2_path)
        if frame2 is None:
            raise ValueError(f"Failed to read image from path: {frame2_path}")
        
        # Create output directory in the same folder as frame2
        output_dir = os.path.dirname(frame2_path)
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur1 = cv2.GaussianBlur(gray1, (blur_size, blur_size), 0)
        blur2 = cv2.GaussianBlur(gray2, (blur_size, blur_size), 0)
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(blur1, blur2)
        
        # Apply threshold
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Dilate to fill in holes
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and collect motion data
        motion_data = []
        
        # Copy frames for visualization
        diff_visualization = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        frame2_copy = frame2.copy()
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                motion_data.append({
                    "index": i,
                    "area": float(area),
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "center": (int(x + w/2), int(y + h/2))
                })
                
                # Draw contour if requested
                if draw:
                    cv2.rectangle(frame2_copy, (x, y), (x+w, y+h), color, thickness)
                    cv2.drawContours(diff_visualization, [contour], -1, color, thickness)
        
        # Save the results to the same directory as input frames
        # Generate filenames based on input frames
        base_name2 = os.path.basename(frame2_path)
        name_parts2 = os.path.splitext(base_name2)
        
        diff_filename = f"{name_parts2[0]}_motion_diff{name_parts2[1]}"
        result_filename = f"{name_parts2[0]}_motion_detected{name_parts2[1]}"
        
        diff_path = os.path.join(output_dir, diff_filename)
        result_path = os.path.join(output_dir, result_filename)
        
        # Save images
        cv2.imwrite(diff_path, diff_visualization)
        cv2.imwrite(result_path, frame2_copy)
        
        return {
            "motion_detected": len(motion_data) > 0,
            "motion_count": len(motion_data),
            "motion_areas": motion_data,
            "total_motion_area": sum(m["area"] for m in motion_data),
            "parameters": {
                "threshold": threshold,
                "blur_size": blur_size,
                "dilate_size": dilate_size,
                "min_area": min_area
            },
            "diff_path": diff_path,
            "path": result_path,
            "output_path": result_path  # Return path for chaining operations
        }
        
    except Exception as e:
        logger.error(f"Error detecting motion: {str(e)}")
        raise ValueError(f"Failed to detect motion: {str(e)}")

def track_object_tool(
    video_path: str,
    initial_bbox: List[int] = None,
    tracker_type: str = "kcf",
    start_frame: int = 0,
    max_frames: int = 50,
    frame_step: int = 1,
    extract_frames: bool = True,
    max_extract: int = 10,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Track an object across video frames
    
    Args:
        video_path: Path to video file
        initial_bbox: Initial bounding box [x, y, width, height]
        tracker_type: Type of tracker ('kcf', 'csrt', 'mil', 'mosse', etc.)
        start_frame: Starting frame index (0-based)
        max_frames: Maximum number of frames to track
        frame_step: Step size (process every nth frame)
        extract_frames: Whether to extract frames with tracking visualization
        max_extract: Maximum number of frames to extract
        output_dir: Directory to save extracted frames (if None, use auto-generated folder)
        
    Returns:
        Dict: Tracking results and extracted frames
    """
    try:
        # Validate video file
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Get video info
        video_info = get_video_info(video_path)
        total_frames = video_info["frame_count"]
        
        # Validate parameters
        if start_frame < 0:
            start_frame = 0
        
        if frame_step < 1:
            frame_step = 1
        
        # Set output directory
        if output_dir is None:
            # Create a dedicated folder for this tracking operation
            output_dir = get_video_output_folder(video_path, "tracking")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create tracker
        tracker_types = {
            "kcf": cv2.legacy.TrackerKCF_create if hasattr(cv2, 'legacy') else cv2.TrackerKCF_create,
            "csrt": cv2.legacy.TrackerCSRT_create if hasattr(cv2, 'legacy') else cv2.TrackerCSRT_create,
            "mil": cv2.legacy.TrackerMIL_create if hasattr(cv2, 'legacy') else cv2.TrackerMIL_create,
            "mosse": cv2.legacy.TrackerMOSSE_create if hasattr(cv2, 'legacy') else cv2.TrackerMOSSE_create,
            "medianflow": cv2.legacy.TrackerMedianFlow_create if hasattr(cv2, 'legacy') else cv2.TrackerMedianFlow_create,
            "tld": cv2.legacy.TrackerTLD_create if hasattr(cv2, 'legacy') else cv2.TrackerTLD_create,
            "boosting": cv2.legacy.TrackerBoosting_create if hasattr(cv2, 'legacy') else cv2.TrackerBoosting_create
        }
        
        if tracker_type.lower() not in tracker_types:
            raise ValueError(f"Unsupported tracker type: {tracker_type}. " + 
                           f"Supported types: {', '.join(tracker_types.keys())}")
        
        tracker_create = tracker_types[tracker_type.lower()]
        
        # Check if we need to select initial bbox from first frame
        if initial_bbox is None:
            # Set position to start_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Read the frame
            ret, frame = cap.read()
            
            if not ret:
                raise ValueError(f"Failed to read frame at index {start_frame}")
            
            # For API use, we'll need the user to provide the initial bbox
            # since we can't open interactive windows for selection
            raise ValueError("initial_bbox must be provided as [x, y, width, height]")
        
        # Validate initial bbox
        if len(initial_bbox) != 4:
            raise ValueError("initial_bbox must contain exactly 4 values: [x, y, width, height]")
        
        # Initialize tracker data
        tracking_data = []
        extracted_frames = []
        
        # Set position to start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read the first frame
        ret, frame = cap.read()
        
        if not ret:
            raise ValueError(f"Failed to read frame at index {start_frame}")
        
        # Initialize tracker
        tracker = tracker_create()
        bbox = tuple(initial_bbox)
        success = tracker.init(frame, bbox)
        
        if not success:
            raise ValueError(f"Failed to initialize tracker with the provided bounding box")
        
        # Add initial tracking data
        tracking_data.append({
            "frame": start_frame,
            "bbox": list(bbox),
            "success": success
        })
        
        # Extract first frame if requested
        if extract_frames:
            # Draw bounding box
            x, y, w, h = [int(v) for v in bbox]
            frame_vis = frame.copy()
            cv2.rectangle(frame_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Save frame to file
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_filename = f"{video_name}_track_{start_frame}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame_vis)
            
            # Add frame data
            frame_timestamp = start_frame / video_info["fps"]
            extracted_frames.append({
                "index": start_frame,
                "timestamp_seconds": float(frame_timestamp),
                "timestamp_formatted": f"{int(frame_timestamp // 60):02d}:{int(frame_timestamp % 60):02d}.{int((frame_timestamp % 1) * 100):02d}",
                "path": frame_path
            })
        
        # Process subsequent frames
        frame_count = 1
        extract_count = 1
        current_frame = start_frame + frame_step
        
        while frame_count < max_frames and current_frame < total_frames:
            # Set position to current frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
            # Read the frame
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Update tracker
            success, bbox = tracker.update(frame)
            
            # Add tracking data
            tracking_data.append({
                "frame": current_frame,
                "bbox": list(bbox) if success else None,
                "success": success
            })
            
            # Extract frame if requested
            if extract_frames and extract_count < max_extract:
                frame_vis = frame.copy()
                
                if success:
                    # Draw bounding box
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Save frame to file
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                frame_filename = f"{video_name}_track_{current_frame}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                
                cv2.imwrite(frame_path, frame_vis)
                
                # Add frame data
                frame_timestamp = current_frame / video_info["fps"]
                extracted_frames.append({
                    "index": current_frame,
                    "timestamp_seconds": float(frame_timestamp),
                    "timestamp_formatted": f"{int(frame_timestamp // 60):02d}:{int(frame_timestamp % 60):02d}.{int((frame_timestamp % 1) * 100):02d}",
                    "tracking_success": success,
                    "path": frame_path
                })
                
                extract_count += 1
            
            # Increment counters
            frame_count += 1
            current_frame += frame_step
        
        # Release the capture
        cap.release()
        
        return {
            "tracking_data": tracking_data,
            "tracked_frame_count": len(tracking_data),
            "successful_tracks": sum(1 for t in tracking_data if t["success"]),
            "extracted_frames": extracted_frames,
            "extracted_frame_count": len(extracted_frames),
            "tracking_parameters": {
                "tracker_type": tracker_type,
                "initial_bbox": initial_bbox,
                "start_frame": start_frame,
                "max_frames": max_frames,
                "frame_step": frame_step
            },
            "video_info": video_info,
            "output_dir": output_dir
        }
        
    except Exception as e:
        logger.error(f"Error tracking object: {str(e)}")
        raise ValueError(f"Failed to track object: {str(e)}")

def combine_frames_to_video_tool(
    frame_paths: List[str],
    output_path: str,
    fps: float = 30.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fourcc: str = "mp4v"
) -> Dict[str, Any]:
    """
    Combine frames into a video file
    
    Args:
        frame_paths: List of paths to frames
        output_path: Path to save the output video
        fps: Frames per second
        width: Output width (if None, use first frame's width)
        height: Output height (if None, use first frame's height)
        fourcc: FourCC code for output video codec
        
    Returns:
        Dict: Video creation results
    """
    try:
        if not frame_paths:
            raise ValueError("No frames provided")
        
        # Get dimensions from first frame if not specified
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            raise ValueError(f"Failed to read first frame from path: {frame_paths[0]}")
        
        if width is None or height is None:
            h, w = first_frame.shape[:2]
            width = width or w
            height = height or h
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create video writer
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        out = cv2.VideoWriter(output_path, fourcc_code, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Failed to create video writer: {output_path}")
        
        # Process each frame
        for i, frame_path in enumerate(frame_paths):
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"Failed to read frame from path: {frame_path}")
                continue
            
            # Resize if necessary
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            # Write frame
            out.write(frame)
        
        # Release writer
        out.release()
        
        return {
            "success": True,
            "output_path": output_path,
            "frame_count": len(frame_paths),
            "video_parameters": {
                "width": width,
                "height": height,
                "fps": fps,
                "fourcc": fourcc
            },
            "size_bytes": os.path.getsize(output_path) if os.path.exists(output_path) else None
        }
        
    except Exception as e:
        logger.error(f"Error combining frames to video: {str(e)}")
        raise ValueError(f"Failed to combine frames to video: {str(e)}")

def create_mp4_from_video_tool(
    video_path: str,
    output_path: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    quality: str = "medium"
) -> Dict[str, Any]:
    """
    Detect a video and create an MP4 file
    
    Args:
        video_path: Path to input video file
        output_path: Path to save the output MP4 (if None, auto-generate)
        width: Output width (if None, use original width)
        height: Output height (if None, use original height)
        fps: Frames per second (if None, use original fps)
        quality: Output quality ('low', 'medium', 'high')
        
    Returns:
        Dict: Conversion results including MP4 path
    """
    try:
        # Validate video file
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
            
        # Check if it's a valid video file
        if not detect_video_file(video_path):
            raise ValueError(f"Not a valid video file: {video_path}")
            
        # Get video info
        video_info = get_video_info(video_path)
        
        # Generate output path if not provided
        if output_path is None:
            video_dir = os.path.dirname(video_path) or '.'
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = get_timestamp()
            output_path = os.path.join(video_dir, f"{video_name}_mp4_{timestamp}.mp4")
        
        # Set dimensions and fps
        target_width = width or video_info["width"]
        target_height = height or video_info["height"]
        target_fps = fps or video_info["fps"]
        
        # Set quality parameters
        quality_settings = {
            "low": {"bitrate": "1000k", "crf": "28"},
            "medium": {"bitrate": "2000k", "crf": "23"},
            "high": {"bitrate": "5000k", "crf": "18"}
        }
        
        # Use medium quality as default if invalid quality specified
        quality = quality.lower()
        if quality not in quality_settings:
            quality = "medium"
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
            
        # Determine fourcc
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        
        # Create video writer
        out = cv2.VideoWriter(
            output_path, 
            fourcc, 
            target_fps, 
            (target_width, target_height)
        )
        
        if not out.isOpened():
            raise ValueError(f"Failed to create video writer: {output_path}")
            
        # Process frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize if necessary
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame = cv2.resize(frame, (target_width, target_height))
                
            # Write frame
            out.write(frame)
            frame_count += 1
            
        # Release resources
        cap.release()
        out.release()
        
        # Verify the output file exists and has valid size
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError(f"Failed to create MP4 file: {output_path}")
            
        # Open the video with system's default player
        open_video_with_system_viewer(output_path)
            
        return {
            "success": True,
            "output_path": output_path,
            "input_path": video_path,
            "frame_count": frame_count,
            "video_parameters": {
                "width": target_width,
                "height": target_height,
                "fps": target_fps,
                "quality": quality
            },
            "size_bytes": os.path.getsize(output_path),
            "size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2)
        }
        
    except Exception as e:
        logger.error(f"Error creating MP4 from video: {str(e)}")
        raise ValueError(f"Failed to create MP4 from video: {str(e)}")

def detect_video_objects_tool(
    video_path: str,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    classes_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    width: int = 416,
    height: int = 416,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    frame_step: int = 1,
    fps: Optional[float] = None,
    show_labels: bool = True,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> Dict[str, Any]:
    """
    Detect objects in a video and create a detection result video
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the output video (if None, auto-generate)
        model_path: Path to model weights file (e.g., yolo.weights)
        config_path: Path to model configuration file (e.g., yolo.cfg)
        classes_path: Path to text file containing class names
        confidence_threshold: Minimum confidence threshold for detections
        nms_threshold: Non-maximum suppression threshold
        width: Network input width
        height: Network input height
        start_frame: Starting frame index (0-based)
        end_frame: Ending frame index (inclusive), if None processes until the end
        frame_step: Step size (process every nth frame)
        fps: Frames per second for output video (if None, use original fps)
        show_labels: Whether to show class labels
        color: Color for drawing boxes (BGR format)
        thickness: Thickness of the bounding box
        
    Returns:
        Dict: Object detection results and output video path
    """
    try:
        # Import platform for video player opening
        import platform
        import subprocess
        
        # Validate video file
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Get video info
        video_info = get_video_info(video_path)
        total_frames = video_info["frame_count"]
        
        # Validate and adjust parameters
        if start_frame < 0:
            start_frame = 0
        
        if end_frame is None or end_frame >= total_frames:
            end_frame = total_frames - 1
        
        if frame_step < 1:
            frame_step = 1
        
        # Generate output path if not provided
        if output_path is None:
            video_dir = os.path.dirname(video_path) or '.'
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = get_timestamp()
            output_path = os.path.join(video_dir, f"{video_name}_detected_{timestamp}.mp4")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set default model directory
        model_dir = os.environ.get("OPENCV_DNN_MODELS_DIR", "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        # Use default YOLO model if paths not provided
        if model_path is None:
            model_path = os.path.join(model_dir, "yolov3.weights")
        
        if config_path is None:
            config_path = os.path.join(model_dir, "yolov3.cfg")
            
        if classes_path is None:
            classes_path = os.path.join(model_dir, "coco.names")
        
        # Check if model files exist
        model_files_exist = os.path.exists(model_path) and os.path.exists(config_path)
        
        if not model_files_exist:
            download_instructions = (
                f"YOLO model files not found at {model_dir}.\n"
                "To download the required files:\n"
                "1. Download YOLOv3 weights file (237MB) from: https://pjreddie.com/media/files/yolov3.weights\n"
                "2. Download YOLOv3 config file from: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg\n"
                "3. Download COCO class names file from: https://github.com/pjreddie/darknet/blob/master/data/coco.names\n"
                "4. Save all files to: {}\n"
                "   OR set the OPENCV_DNN_MODELS_DIR environment variable to your preferred directory\n"
                "5. Restart the application".format(model_dir)
            )
            logger.warning(download_instructions)
            return {
                "error": "YOLO model files not found",
                "download_instructions": download_instructions,
                "model_paths": {
                    "model_path": model_path,
                    "config_path": config_path,
                    "classes_path": classes_path
                }
            }
        
        # Load class names
        try:
            with open(classes_path, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
        except Exception as e:
            logger.error(f"Error loading class names: {str(e)}")
            # Provide a small subset of COCO classes as fallback
            classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
        
        # Try to load the model
        try:
            net = cv2.dnn.readNetFromDarknet(config_path, model_path)
        except Exception as e:
            logger.error(f"Error loading DNN model: {str(e)}")
            return {
                "error": f"Failed to load DNN model: {str(e)}",
                "model_paths": {
                    "model_path": model_path,
                    "config_path": config_path
                }
            }
        
        # Get output layer names
        layer_names = net.getLayerNames()
        
        # OpenCV 4.5.4+ has a different indexing system
        try:
            # For newer OpenCV versions
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            # For older OpenCV versions
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        # Initialize video writer
        target_fps = fps or video_info["fps"]
        target_width = int(video_info["width"])
        target_height = int(video_info["height"])
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            target_fps,
            (target_width, target_height)
        )
        
        if not out.isOpened():
            raise ValueError(f"Failed to create video writer: {output_path}")
        
        # Initialize counters and results
        frame_count = 0
        processed_count = 0
        detection_counts = {}
        
        # Set position to start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        
        # Process frames
        while current_frame <= end_frame:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame for object detection
            frame_vis = frame.copy()
            orig_h, orig_w = frame.shape[:2]
            
            # Create a blob from the frame
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (width, height), swapRB=True, crop=False)
            
            # Set the input to the network
            net.setInput(blob)
            
            # Run forward pass
            layer_outputs = net.forward(output_layers)
            
            # Initialize lists for detected objects
            boxes = []
            confidences = []
            class_ids = []
            
            # Process each output layer
            for output in layer_outputs:
                # Process each detection
                for detection in output:
                    # The first 4 elements are bounding box coordinates
                    # The rest are class probabilities
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Filter by confidence threshold
                    if confidence > confidence_threshold:
                        # Get bounding box coordinates
                        # YOLO returns center (x, y) and width, height
                        center_x = int(detection[0] * orig_w)
                        center_y = int(detection[1] * orig_h)
                        width_box = int(detection[2] * orig_w)
                        height_box = int(detection[3] * orig_h)
                        
                        # Calculate top-left corner
                        x = int(center_x - width_box / 2)
                        y = int(center_y - height_box / 2)
                        
                        # Add to lists
                        boxes.append([x, y, width_box, height_box])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            
            # Draw bounding boxes on the frame
            if len(indices) > 0:
                # Ensure indices is properly formatted (OpenCV 4.5.4+ compatibility)
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                
                # Process each selected box
                for i in indices:
                    # Extract bounding box coordinates
                    # In some versions this may be i[0] instead of just i
                    idx = i if isinstance(i, int) else i[0]
                    
                    box = boxes[idx]
                    x, y, w, h = box
                    
                    # Get class name and confidence
                    class_id = class_ids[idx]
                    class_name = classes[class_id] if class_id < len(classes) else f"Class {class_id}"
                    confidence = confidences[idx]
                    
                    # Update detection counts
                    detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    x_end = min(orig_w, x + w)
                    y_end = min(orig_h, y + h)
                    
                    # Draw rectangle
                    cv2.rectangle(frame_vis, (x, y), (x_end, y_end), color, thickness)
                    
                    # Add label
                    if show_labels:
                        text = f"{class_name}: {confidence:.2f}"
                        y_text = y - 10 if y - 10 > 10 else y + 10
                        cv2.putText(frame_vis, text, (x, y_text),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Write frame to output video
            out.write(frame_vis)
            
            # Update counters
            frame_count += 1
            processed_count += 1
            
            # Skip to next frame
            current_frame += frame_step
            if current_frame <= end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        # Release resources
        cap.release()
        out.release()
        
        # Open the output video with system's default player
        open_video_with_system_viewer(output_path)
        
        # Return results
        return {
            "success": True,
            "output_path": output_path,
            "input_path": video_path,
            "processed_frames": processed_count,
            "detection_counts": detection_counts,
            "total_detections": sum(detection_counts.values()),
            "detection_parameters": {
                "confidence_threshold": confidence_threshold,
                "nms_threshold": nms_threshold,
                "input_size": (width, height)
            },
            "model_info": {
                "model_path": model_path,
                "config_path": config_path,
                "classes_count": len(classes)
            },
            "video_parameters": {
                "width": target_width,
                "height": target_height,
                "fps": target_fps
            },
            "size_bytes": os.path.getsize(output_path) if os.path.exists(output_path) else None,
            "size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2) if os.path.exists(output_path) else None
        }
        
    except Exception as e:
        logger.error(f"Error detecting video objects: {str(e)}")
        raise ValueError(f"Failed to detect video objects: {str(e)}")

def detect_camera_objects_tool(
    camera_id: int = 0,
    duration: int = 30,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    classes_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    width: int = 416,
    height: int = 416,
    fps: int = 30,
    show_labels: bool = True,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> Dict[str, Any]:
    """
    Detect objects from computer's camera and save to video
    
    Args:
        camera_id: Camera device ID (0 for default camera)
        duration: Duration to record in seconds
        output_path: Path to save the output video (if None, auto-generate)
        model_path: Path to model weights file (e.g., yolo.weights)
        config_path: Path to model configuration file (e.g., yolo.cfg)
        classes_path: Path to text file containing class names
        confidence_threshold: Minimum confidence threshold for detections
        nms_threshold: Non-maximum suppression threshold
        width: Network input width
        height: Network input height
        fps: Frames per second for output video
        show_labels: Whether to show class labels
        color: Color for drawing boxes (BGR format)
        thickness: Thickness of the bounding box
        
    Returns:
        Dict: Object detection results and output video path
    """
    try:
        # Import platform for video player opening
        import platform
        import subprocess
        import time
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = get_timestamp()
            output_path = os.path.join(os.getcwd(), f"camera_detected_{timestamp}.mp4")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Open the camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open camera with ID {camera_id}")
        
        # Read first frame to get dimensions
        ret, frame = cap.read()
        
        if not ret:
            raise ValueError("Failed to read frame from camera")
        
        # Get camera dimensions
        camera_height, camera_width = frame.shape[:2]
        
        # Set default model directory
        model_dir = os.environ.get("OPENCV_DNN_MODELS_DIR", "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        # Use default YOLO model if paths not provided
        if model_path is None:
            model_path = os.path.join(model_dir, "yolov3.weights")
        
        if config_path is None:
            config_path = os.path.join(model_dir, "yolov3.cfg")
            
        if classes_path is None:
            classes_path = os.path.join(model_dir, "coco.names")
        
        # Check if model files exist
        model_files_exist = os.path.exists(model_path) and os.path.exists(config_path)
        
        if not model_files_exist:
            download_instructions = (
                f"YOLO model files not found at {model_dir}.\n"
                "To download the required files:\n"
                "1. Download YOLOv3 weights file (237MB) from: https://pjreddie.com/media/files/yolov3.weights\n"
                "2. Download YOLOv3 config file from: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg\n"
                "3. Download COCO class names file from: https://github.com/pjreddie/darknet/blob/master/data/coco.names\n"
                "4. Save all files to: {}\n"
                "   OR set the OPENCV_DNN_MODELS_DIR environment variable to your preferred directory\n"
                "5. Restart the application".format(model_dir)
            )
            logger.warning(download_instructions)
            return {
                "error": "YOLO model files not found",
                "download_instructions": download_instructions,
                "model_paths": {
                    "model_path": model_path,
                    "config_path": config_path,
                    "classes_path": classes_path
                }
            }
        
        # Load class names
        try:
            with open(classes_path, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
        except Exception as e:
            logger.error(f"Error loading class names: {str(e)}")
            # Provide a small subset of COCO classes as fallback
            classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
        
        # Try to load the model
        try:
            net = cv2.dnn.readNetFromDarknet(config_path, model_path)
        except Exception as e:
            logger.error(f"Error loading DNN model: {str(e)}")
            return {
                "error": f"Failed to load DNN model: {str(e)}",
                "model_paths": {
                    "model_path": model_path,
                    "config_path": config_path
                }
            }
        
        # Get output layer names
        layer_names = net.getLayerNames()
        
        # OpenCV 4.5.4+ has a different indexing system
        try:
            # For newer OpenCV versions
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            # For older OpenCV versions
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (camera_width, camera_height)
        )
        
        if not out.isOpened():
            raise ValueError(f"Failed to create video writer: {output_path}")
        
        # Initialize counters and results
        frame_count = 0
        detection_counts = {}
        start_time = time.time()
        
        # Process frames for the specified duration
        while time.time() - start_time < duration:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame for object detection
            frame_vis = frame.copy()
            orig_h, orig_w = frame.shape[:2]
            
            # Create a blob from the frame
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (width, height), swapRB=True, crop=False)
            
            # Set the input to the network
            net.setInput(blob)
            
            # Run forward pass
            layer_outputs = net.forward(output_layers)
            
            # Initialize lists for detected objects
            boxes = []
            confidences = []
            class_ids = []
            
            # Process each output layer
            for output in layer_outputs:
                # Process each detection
                for detection in output:
                    # The first 4 elements are bounding box coordinates
                    # The rest are class probabilities
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Filter by confidence threshold
                    if confidence > confidence_threshold:
                        # Get bounding box coordinates
                        # YOLO returns center (x, y) and width, height
                        center_x = int(detection[0] * orig_w)
                        center_y = int(detection[1] * orig_h)
                        width_box = int(detection[2] * orig_w)
                        height_box = int(detection[3] * orig_h)
                        
                        # Calculate top-left corner
                        x = int(center_x - width_box / 2)
                        y = int(center_y - height_box / 2)
                        
                        # Add to lists
                        boxes.append([x, y, width_box, height_box])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            
            # Draw bounding boxes on the frame
            if len(indices) > 0:
                # Ensure indices is properly formatted (OpenCV 4.5.4+ compatibility)
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                
                # Process each selected box
                for i in indices:
                    # Extract bounding box coordinates
                    # In some versions this may be i[0] instead of just i
                    idx = i if isinstance(i, int) else i[0]
                    
                    box = boxes[idx]
                    x, y, w, h = box
                    
                    # Get class name and confidence
                    class_id = class_ids[idx]
                    class_name = classes[class_id] if class_id < len(classes) else f"Class {class_id}"
                    confidence = confidences[idx]
                    
                    # Update detection counts
                    detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    x_end = min(orig_w, x + w)
                    y_end = min(orig_h, y + h)
                    
                    # Draw rectangle
                    cv2.rectangle(frame_vis, (x, y), (x_end, y_end), color, thickness)
                    
                    # Add label
                    if show_labels:
                        text = f"{class_name}: {confidence:.2f}"
                        y_text = y - 10 if y - 10 > 10 else y + 10
                        cv2.putText(frame_vis, text, (x, y_text),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add recording information
            elapsed_time = time.time() - start_time
            remaining_time = max(0, duration - elapsed_time)
            cv2.putText(frame_vis, f"Recording: {int(remaining_time)}s remaining", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Write frame to output video
            out.write(frame_vis)
            
            # Update counters
            frame_count += 1
            
            # Display the frame (optional)
            cv2.imshow("Camera Detection", frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Open the output video with system's default player
        open_video_with_system_viewer(output_path)
        
        # Return results
        return {
            "success": True,
            "output_path": output_path,
            "frames_captured": frame_count,
            "detection_counts": detection_counts,
            "total_detections": sum(detection_counts.values()),
            "detection_parameters": {
                "confidence_threshold": confidence_threshold,
                "nms_threshold": nms_threshold,
                "input_size": (width, height)
            },
            "model_info": {
                "model_path": model_path,
                "config_path": config_path,
                "classes_count": len(classes)
            },
            "video_parameters": {
                "width": camera_width,
                "height": camera_height,
                "fps": fps,
                "duration": duration
            },
            "size_bytes": os.path.getsize(output_path) if os.path.exists(output_path) else None,
            "size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2) if os.path.exists(output_path) else None
        }
        
    except Exception as e:
        logger.error(f"Error detecting camera objects: {str(e)}")
        raise ValueError(f"Failed to detect camera objects: {str(e)}")

def register_tools(mcp):
    """
    Register all video processing tools with the MCP server
    
    Args:
        mcp: The MCP server instance
    """
    # Register tool implementations
    mcp.add_tool(extract_video_frames_tool)
    mcp.add_tool(detect_motion_tool)
    mcp.add_tool(track_object_tool)
    mcp.add_tool(combine_frames_to_video_tool)
    mcp.add_tool(create_mp4_from_video_tool)
    mcp.add_tool(detect_video_objects_tool)
    mcp.add_tool(detect_camera_objects_tool)
