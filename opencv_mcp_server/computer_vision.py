"""
OpenCV MCP Server - Computer Vision

This module provides high-level computer vision and object detection tools using OpenCV.
It includes functionality for feature detection and matching, face detection,
and object detection using pre-trained models.
"""

import cv2
import numpy as np
import os
import logging
from typing import Optional, List, Dict, Any, Tuple, Union

# Import utility functions from utils
from .utils import get_image_info, save_and_display, get_timestamp

logger = logging.getLogger("opencv-mcp-server.computer_vision")

# Tool implementations
def detect_features_tool(
    image_path: str, 
    method: str = "sift", 
    max_features: int = 500,
    draw: bool = True,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> Dict[str, Any]:
    """
    Detect features in an image
    
    Args:
        image_path: Path to the image file
        method: Feature detection method ('sift', 'orb', 'brisk', 'akaze')
        max_features: Maximum number of features to detect
        draw: Whether to draw keypoints on the image
        color: Color for drawing keypoints (BGR format)
        
    Returns:
        Dict: Image with keypoints and feature information
    """
    try:
        # Read image from path
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Initialize the feature detector
        if method.lower() == 'sift':
            detector = cv2.SIFT_create(nfeatures=max_features)
        elif method.lower() == 'orb':
            detector = cv2.ORB_create(nfeatures=max_features)
        elif method.lower() == 'brisk':
            detector = cv2.BRISK_create()
        elif method.lower() == 'akaze':
            detector = cv2.AKAZE_create()
        else:
            raise ValueError(f"Unsupported feature detection method: {method}")
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        
        # Draw keypoints if requested
        if draw:
            img_keypoints = cv2.drawKeypoints(
                img, 
                keypoints, 
                None, 
                color=color, 
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
        else:
            img_keypoints = img.copy()
        
        # Prepare keypoint data
        keypoint_data = []
        for i, kp in enumerate(keypoints[:50]):  # Limit to 50 for response size
            keypoint_data.append({
                "index": i,
                "x": float(kp.pt[0]),
                "y": float(kp.pt[1]),
                "size": float(kp.size),
                "angle": float(kp.angle),
                "response": float(kp.response),
                "octave": int(kp.octave)
            })
        
        # Save and display
        new_path = save_and_display(img_keypoints, image_path, f"features_{method}")
        
        return {
            "keypoint_count": len(keypoints),
            "keypoints": keypoint_data,
            "method": method,
            "info": get_image_info(img_keypoints),
            "path": new_path,
            "output_path": new_path  # Return path for chaining operations
        }
        
    except Exception as e:
        logger.error(f"Error detecting features: {str(e)}")
        raise ValueError(f"Failed to detect features: {str(e)}")

def match_features_tool(
    image1_path: str, 
    image2_path: str, 
    method: str = "sift",
    matcher: str = "bf",
    max_features: int = 500,
    ratio_thresh: float = 0.75,
    draw: bool = True
) -> Dict[str, Any]:
    """
    Match features between two images
    
    Args:
        image1_path: Path to the first image file
        image2_path: Path to the second image file
        method: Feature detection method ('sift', 'orb', 'brisk', 'akaze')
        matcher: Matcher type ('bf' for BFMatcher, 'flann' for FlannBasedMatcher)
        max_features: Maximum number of features to detect
        ratio_thresh: Ratio threshold for Lowe's ratio test
        draw: Whether to draw matches
        
    Returns:
        Dict: Image with matches and match information
    """
    try:
        # Read images from paths
        img1 = cv2.imread(image1_path)
        if img1 is None:
            raise ValueError(f"Failed to read image from path: {image1_path}")
            
        img2 = cv2.imread(image2_path)
        if img2 is None:
            raise ValueError(f"Failed to read image from path: {image2_path}")
        
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Initialize the feature detector
        if method.lower() == 'sift':
            detector = cv2.SIFT_create(nfeatures=max_features)
            is_binary = False
        elif method.lower() == 'orb':
            detector = cv2.ORB_create(nfeatures=max_features)
            is_binary = True
        elif method.lower() == 'brisk':
            detector = cv2.BRISK_create()
            is_binary = True
        elif method.lower() == 'akaze':
            detector = cv2.AKAZE_create()
            is_binary = True
        else:
            raise ValueError(f"Unsupported feature detection method: {method}")
        
        # Detect keypoints and compute descriptors
        keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
        
        # Check if descriptors were found
        if descriptors1 is None or descriptors2 is None or len(keypoints1) == 0 or len(keypoints2) == 0:
            return {
                "match_count": 0,
                "error": "No keypoints or descriptors found in one or both images",
                "keypoint_count1": len(keypoints1) if keypoints1 else 0,
                "keypoint_count2": len(keypoints2) if keypoints2 else 0,
            }
        
        # Create feature matcher
        if matcher.lower() == 'bf':
            if is_binary:
                norm_type = cv2.NORM_HAMMING
            else:
                norm_type = cv2.NORM_L2
                
            match_obj = cv2.BFMatcher(norm_type, crossCheck=False)
            
        elif matcher.lower() == 'flann':
            if is_binary:
                # Special FLANN params for binary descriptors
                flann_params = dict(
                    algorithm=6,  # FLANN_INDEX_LSH
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1
                )
            else:
                flann_params = dict(
                    algorithm=1,  # FLANN_INDEX_KDTREE
                    trees=5
                )
                
            match_obj = cv2.FlannBasedMatcher(flann_params, {})
        else:
            raise ValueError(f"Unsupported matcher type: {matcher}")
        
        # Apply ratio test if not using crossCheck
        matches = match_obj.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        
        # Collect match data
        match_data = []
        for i, m in enumerate(good_matches[:50]):  # Limit to 50 for response size
            match_data.append({
                "index": i,
                "queryIdx": m.queryIdx,
                "trainIdx": m.trainIdx,
                "distance": float(m.distance),
                "image1_point": (float(keypoints1[m.queryIdx].pt[0]), float(keypoints1[m.queryIdx].pt[1])),
                "image2_point": (float(keypoints2[m.trainIdx].pt[0]), float(keypoints2[m.trainIdx].pt[1]))
            })
        
        # Draw matches if requested
        if draw and good_matches:
            draw_params = dict(
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 0),
                flags=cv2.DrawMatchesFlags_DEFAULT
            )
            
            # Only draw a limited number of matches
            draw_matches = good_matches[:50] if len(good_matches) > 50 else good_matches
            
            img_matches = cv2.drawMatches(
                img1, keypoints1, 
                img2, keypoints2, 
                draw_matches, None, 
                **draw_params
            )
        else:
            # Create a side-by-side image
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            h = max(h1, h2)
            img_matches = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
            img_matches[:h1, :w1] = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img_matches[:h2, w1:w1+w2] = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # Generate a unique output path based on both input images
        base_name1 = os.path.basename(image1_path)
        base_name2 = os.path.basename(image2_path)
        name_parts1 = os.path.splitext(base_name1)
        directory = os.path.dirname(image1_path) or '.'
        
        # Save and display
        result_path = save_and_display(img_matches, image1_path, f"matches_{method}_{name_parts1[0]}_to_{os.path.splitext(base_name2)[0]}")
        
        return {
            "match_count": len(good_matches),
            "matches": match_data,
            "keypoint_count1": len(keypoints1),
            "keypoint_count2": len(keypoints2),
            "match_parameters": {
                "method": method,
                "matcher": matcher,
                "ratio_thresh": ratio_thresh
            },
            "info": get_image_info(img_matches),
            "path": result_path,
            "output_path": result_path  # Return path for chaining operations
        }
        
    except Exception as e:
        logger.error(f"Error matching features: {str(e)}")
        raise ValueError(f"Failed to match features: {str(e)}")

def detect_faces_tool(
    image_path: str, 
    method: str = "haar", 
    scale_factor: float = 1.3,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (30, 30),
    confidence_threshold: float = 0.5,
    draw: bool = True,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> Dict[str, Any]:
    """
    Detect faces in an image
    
    Args:
        image_path: Path to the image file
        method: Face detection method ('haar', 'dnn')
        scale_factor: Scale factor for cascade classifier
        min_neighbors: Minimum neighbors for cascade classifier
        min_size: Minimum face size to detect
        confidence_threshold: Confidence threshold for DNN method
        draw: Whether to draw rectangles around faces
        color: Color for drawing rectangles (BGR format)
        thickness: Thickness of the rectangle
        
    Returns:
        Dict: Image with faces and face information
    """
    try:
        # Read image from path
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        
        # Make a copy for drawing
        img_copy = img.copy()
        
        # Get face cascade directory
        cascade_path = cv2.data.haarcascades
        haar_xml = os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
        
        # Check if cascade file exists
        if not os.path.exists(haar_xml):
            download_instructions = (
                f"Haar cascade file not found: {haar_xml}\n"
                "To download the files:\n"
                "1. Download from: https://github.com/opencv/opencv/tree/master/data/haarcascades\n"
                "2. Save to the OpenCV data directory at: {}\n"
                "   OR set the CV_HAAR_CASCADE_DIR environment variable to your preferred location\n"
                "3. Restart the application".format(cascade_path)
            )
            logger.error(download_instructions)
            raise FileNotFoundError(download_instructions)
        
        faces = []
        
        if method.lower() == 'haar':
            # Load the face detector
            face_cascade = cv2.CascadeClassifier(haar_xml)
            
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Detect faces
            faces_rect = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )
            
            # Prepare face data
            for i, (x, y, w, h) in enumerate(faces_rect):
                faces.append({
                    "index": i,
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "confidence": None  # Haar doesn't provide confidence
                })
                
                # Draw rectangle if requested
                if draw:
                    cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, thickness)
            
        elif method.lower() == 'dnn':
            # Paths to the model files
            # Note: You would need to download these files for actual use
            model_dir = os.environ.get("OPENCV_DNN_MODELS_DIR", "models")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                
            prototxt_path = os.path.join(model_dir, "deploy.prototxt")
            model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
            
            # Check if model files exist
            model_files_exist = os.path.exists(prototxt_path) and os.path.exists(model_path)
            
            if not model_files_exist:
                download_instructions = (
                    f"DNN model files not found at {model_dir}.\n"
                    "To download the required files:\n"
                    "1. Download the model files from: https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830\n"
                    "   - deploy.prototxt\n"
                    "   - res10_300x300_ssd_iter_140000.caffemodel\n"
                    "2. Save them to: {}\n"
                    "   OR set the OPENCV_DNN_MODELS_DIR environment variable to your preferred directory\n"
                    "3. Restart the application".format(model_dir)
                )
                logger.warning(download_instructions)
                return {
                    "error": "DNN model files not found",
                    "download_instructions": download_instructions,
                    "face_count": 0,
                    "faces": []
                }
            
            # Try to load the model
            try:
                face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            except Exception as e:
                logger.error(f"Error loading DNN model: {str(e)}")
                return {
                    "error": f"Failed to load DNN model: {str(e)}",
                    "info": {
                        "prototxt_path": prototxt_path,
                        "model_path": model_path
                    }
                }
            
            # Get image dimensions
            (h, w) = img.shape[:2]
            
            # Create a blob from the image
            blob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)), 
                1.0, 
                (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            # Pass the blob through the network
            face_net.setInput(blob)
            detections = face_net.forward()
            
            # Process detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter by confidence threshold
                if confidence > confidence_threshold:
                    # Get coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure coordinates are within image bounds
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    width = endX - startX
                    height = endY - startY
                    
                    faces.append({
                        "index": i,
                        "x": int(startX),
                        "y": int(startY),
                        "width": int(width),
                        "height": int(height),
                        "confidence": float(confidence)
                    })
                    
                    # Draw rectangle if requested
                    if draw:
                        cv2.rectangle(img_copy, (startX, startY), (endX, endY), color, thickness)
                        # Draw confidence text
                        text = f"{confidence:.2f}"
                        y_text = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.putText(img_copy, text, (startX, y_text),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        else:
            raise ValueError(f"Unsupported face detection method: {method}")
        
        # Save and display
        result_path = save_and_display(img_copy, image_path, f"faces_{method}")
        
        return {
            "face_count": len(faces),
            "faces": faces,
            "method": method,
            "info": get_image_info(img_copy),
            "path": result_path,
            "output_path": result_path  # Return path for chaining operations
        }
        
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        raise ValueError(f"Failed to detect faces: {str(e)}")

def detect_objects_tool(
    image_path: str, 
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    classes_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    width: int = 416,
    height: int = 416,
    draw: bool = True,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> Dict[str, Any]:
    """
    Detect objects using pre-trained DNN models
    
    Args:
        image_path: Path to the image file
        model_path: Path to model weights file (e.g., yolo.weights)
        config_path: Path to model configuration file (e.g., yolo.cfg)
        classes_path: Path to text file containing class names
        confidence_threshold: Minimum confidence threshold
        nms_threshold: Non-maximum suppression threshold
        width: Network input width
        height: Network input height
        draw: Whether to draw boxes around objects
        color: Color for drawing boxes (BGR format)
        thickness: Thickness of the box
        
    Returns:
        Dict: Image with objects and object information
    """
    try:
        # Read image from path
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        
        # Make a copy for drawing
        img_copy = img.copy()
        
        # Get image dimensions
        (orig_h, orig_w) = img.shape[:2]
        
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
                "info": {
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
                "info": {
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
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (width, height), swapRB=True, crop=False)
        
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
        
        # Prepare object data
        objects = []
        
        # Check if indices is not empty
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
                
                # Get class name
                class_id = class_ids[idx]
                class_name = classes[class_id] if class_id < len(classes) else f"Class {class_id}"
                
                # Prepare object data
                objects.append({
                    "index": idx,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(confidences[idx]),
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                })
                
                # Draw rectangle if requested
                if draw:
                    # Ensure coordinates are within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    x_end = min(orig_w, x + w)
                    y_end = min(orig_h, y + h)
                    
                    # Draw rectangle
                    cv2.rectangle(img_copy, (x, y), (x_end, y_end), color, thickness)
                    
                    # Add label
                    text = f"{class_name}: {confidences[idx]:.2f}"
                    y_text = y - 10 if y - 10 > 10 else y + 10
                    cv2.putText(img_copy, text, (x, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save and display
        result_path = save_and_display(img_copy, image_path, "objects_detected")
        
        return {
            "object_count": len(objects),
            "objects": objects,
            "model_info": {
                "model_path": model_path,
                "config_path": config_path,
                "classes_count": len(classes),
                "input_size": (width, height)
            },
            "info": get_image_info(img_copy),
            "path": result_path,
            "output_path": result_path  # Return path for chaining operations
        }
        
    except Exception as e:
        logger.error(f"Error detecting objects: {str(e)}")
        raise ValueError(f"Failed to detect objects: {str(e)}")

def register_tools(mcp):
    """
    Register all computer vision tools with the MCP server
    
    Args:
        mcp: The MCP server instance
    """
    # Register tool implementations
    mcp.add_tool(detect_features_tool)
    mcp.add_tool(match_features_tool)
    mcp.add_tool(detect_faces_tool)
    mcp.add_tool(detect_objects_tool)