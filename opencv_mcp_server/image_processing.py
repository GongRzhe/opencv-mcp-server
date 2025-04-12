"""
OpenCV MCP Server - Image Processing

This module provides advanced image processing and transformation tools using OpenCV.
It includes functionality for filtering, edge detection, thresholding, contour detection,
shape finding, and template matching.
"""

import cv2
import numpy as np
import os
import logging
from typing import Optional, List, Dict, Any, Tuple, Union

# Import utility functions from utils
from .utils import get_image_info, save_and_display, get_timestamp

logger = logging.getLogger("opencv-mcp-server.image_processing")

# Tool implementations
def apply_filter_tool(
    image_path: str, 
    filter_type: str, 
    kernel_size: Union[int, Tuple[int, int]], 
    sigma: Optional[float] = None,
    sigma_color: Optional[float] = None,
    sigma_space: Optional[float] = None
) -> Dict[str, Any]:
    """
    Apply various filters to an image
    
    Args:
        image_path: Path to the image file
        filter_type: Type of filter ('blur', 'gaussian', 'median', 'bilateral')
        kernel_size: Size of the kernel, should be odd (e.g., 3, 5, 7)
        sigma: Standard deviation for Gaussian filter
        sigma_color: Filter sigma in the color space for bilateral filter
        sigma_space: Filter sigma in the coordinate space for bilateral filter
        
    Returns:
        Dict: Filtered image and filter information
    """
    try:
        # Read image from path
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        
        # Ensure kernel size is odd
        if isinstance(kernel_size, int):
            if kernel_size % 2 == 0:
                kernel_size = kernel_size + 1
            kernel_size = (kernel_size, kernel_size)
        
        # Apply selected filter
        if filter_type.lower() == 'blur':
            result = cv2.blur(img, kernel_size)
            filter_info = {"type": "blur", "kernel_size": kernel_size}
        
        elif filter_type.lower() == 'gaussian':
            sigma_val = sigma if sigma is not None else 0
            result = cv2.GaussianBlur(img, kernel_size, sigma_val)
            filter_info = {"type": "gaussian", "kernel_size": kernel_size, "sigma": sigma_val}
        
        elif filter_type.lower() == 'median':
            # Median filter takes just an integer for kernel size
            k_size = max(kernel_size[0], kernel_size[1])
            if k_size % 2 == 0:
                k_size += 1
            result = cv2.medianBlur(img, k_size)
            filter_info = {"type": "median", "kernel_size": k_size}
        
        elif filter_type.lower() == 'bilateral':
            s_color = sigma_color if sigma_color is not None else 75
            s_space = sigma_space if sigma_space is not None else 75
            d = max(kernel_size[0], kernel_size[1])
            result = cv2.bilateralFilter(img, d, s_color, s_space)
            filter_info = {
                "type": "bilateral", 
                "diameter": d, 
                "sigma_color": s_color, 
                "sigma_space": s_space
            }
        
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        
        # Save and display
        new_path = save_and_display(result, image_path, f"filter_{filter_type}")
        
        return {
            "filter": filter_info,
            "info": get_image_info(result),
            "path": new_path,
            "output_path": new_path  # Return path for chaining operations
        }
        
    except Exception as e:
        logger.error(f"Error applying filter: {str(e)}")
        raise ValueError(f"Failed to apply filter: {str(e)}")

def detect_edges_tool(
    image_path: str, 
    method: str = "canny", 
    threshold1: float = 100.0, 
    threshold2: float = 200.0,
    aperture_size: int = 3,
    l2gradient: bool = False,
    ksize: int = 3,
    scale: float = 1.0,
    delta: float = 0.0
) -> Dict[str, Any]:
    """
    Detect edges in an image
    
    Args:
        image_path: Path to the image file
        method: Edge detection method ('canny', 'sobel', 'laplacian', 'scharr')
        threshold1: First threshold for Canny detector
        threshold2: Second threshold for Canny detector
        aperture_size: Aperture size for Canny and Laplacian
        l2gradient: Flag for L2 gradient in Canny
        ksize: Kernel size for Sobel and Laplacian
        scale: Scale factor for Sobel, Laplacian, and Scharr
        delta: Delta value added to results for Sobel, Laplacian, and Scharr
        
    Returns:
        Dict: Edge-detected image and method information
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
        
        method_info = {"method": method}
        
        if method.lower() == 'canny':
            edges = cv2.Canny(
                gray, 
                threshold1, 
                threshold2, 
                apertureSize=aperture_size, 
                L2gradient=l2gradient
            )
            method_info.update({
                "threshold1": threshold1,
                "threshold2": threshold2,
                "aperture_size": aperture_size,
                "l2gradient": l2gradient
            })
            
        elif method.lower() == 'sobel':
            # Compute Sobel gradients
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)
            
            # Compute gradient magnitude
            magnitude = cv2.magnitude(sobelx, sobely)
            
            # Normalize to 0-255 and convert to uint8
            edges = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            method_info.update({
                "ksize": ksize,
                "scale": scale,
                "delta": delta
            })
            
        elif method.lower() == 'laplacian':
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)
            edges = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            method_info.update({
                "ksize": ksize,
                "scale": scale,
                "delta": delta
            })
            
        elif method.lower() == 'scharr':
            # Scharr is similar to Sobel but with a fixed 3x3 kernel
            scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale=scale, delta=delta)
            scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale=scale, delta=delta)
            
            # Compute gradient magnitude
            magnitude = cv2.magnitude(scharrx, scharry)
            edges = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            method_info.update({
                "scale": scale,
                "delta": delta
            })
            
        else:
            raise ValueError(f"Unsupported edge detection method: {method}")
        
        # Save and display
        new_path = save_and_display(edges, image_path, f"edges_{method}")
        
        return {
            "method_info": method_info,
            "info": get_image_info(edges),
            "path": new_path,
            "output_path": new_path  # Return path for chaining operations
        }
        
    except Exception as e:
        logger.error(f"Error detecting edges: {str(e)}")
        raise ValueError(f"Failed to detect edges: {str(e)}")

def apply_threshold_tool(
    image_path: str, 
    threshold_type: str = "binary", 
    threshold_value: float = 127.0, 
    max_value: float = 255.0,
    adaptive_method: str = "gaussian",
    block_size: int = 11,
    c: float = 2.0
) -> Dict[str, Any]:
    """
    Apply threshold to an image
    
    Args:
        image_path: Path to the image file
        threshold_type: Type of thresholding ('binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv', 'adaptive')
        threshold_value: Threshold value for global thresholding
        max_value: Maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV types
        adaptive_method: Adaptive method for adaptive thresholding ('mean', 'gaussian')
        block_size: Size of a pixel neighborhood for adaptive thresholding
        c: Constant subtracted from the mean or weighted mean for adaptive thresholding
        
    Returns:
        Dict: Thresholded image and threshold information
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
        
        # Map threshold types to OpenCV constants
        threshold_types = {
            "binary": cv2.THRESH_BINARY,
            "binary_inv": cv2.THRESH_BINARY_INV,
            "trunc": cv2.THRESH_TRUNC,
            "tozero": cv2.THRESH_TOZERO,
            "tozero_inv": cv2.THRESH_TOZERO_INV
        }
        
        # Map adaptive methods to OpenCV constants
        adaptive_methods = {
            "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
            "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        }
        
        threshold_info = {"threshold_type": threshold_type}
        
        if threshold_type.lower() == 'adaptive':
            # Ensure block_size is odd
            if block_size % 2 == 0:
                block_size += 1
                
            # Get adaptive method constant
            if adaptive_method.lower() not in adaptive_methods:
                raise ValueError(f"Unsupported adaptive method: {adaptive_method}")
            method = adaptive_methods[adaptive_method.lower()]
            
            # Apply adaptive threshold
            result = cv2.adaptiveThreshold(
                gray,
                max_value,
                method,
                cv2.THRESH_BINARY,
                block_size,
                c
            )
            
            threshold_info.update({
                "adaptive_method": adaptive_method,
                "block_size": block_size,
                "c": c,
                "max_value": max_value
            })
            
        else:
            # Get threshold type constant
            if threshold_type.lower() not in threshold_types:
                raise ValueError(f"Unsupported threshold type: {threshold_type}")
            thresh_type = threshold_types[threshold_type.lower()]
            
            # Apply global threshold
            _, result = cv2.threshold(gray, threshold_value, max_value, thresh_type)
            
            threshold_info.update({
                "threshold_value": threshold_value,
                "max_value": max_value
            })
        
        # Save and display
        new_path = save_and_display(result, image_path, f"threshold_{threshold_type}")
        
        return {
            "threshold_info": threshold_info,
            "info": get_image_info(result),
            "path": new_path,
            "output_path": new_path  # Return path for chaining operations
        }
        
    except Exception as e:
        logger.error(f"Error applying threshold: {str(e)}")
        raise ValueError(f"Failed to apply threshold: {str(e)}")

def detect_contours_tool(
    image_path: str, 
    mode: str = "external", 
    method: str = "simple",
    draw: bool = True,
    thickness: int = 1,
    color: Tuple[int, int, int] = (0, 255, 0),
    threshold_value: float = 127.0
) -> Dict[str, Any]:
    """
    Detect and optionally draw contours in an image
    
    Args:
        image_path: Path to the image file
        mode: Contour retrieval mode ('external', 'list', 'ccomp', 'tree')
        method: Contour approximation method ('none', 'simple', 'tc89_l1', 'tc89_kcos')
        draw: Whether to draw the contours on the image
        thickness: Thickness of contour lines
        color: Color for drawing contours (BGR format)
        threshold_value: Threshold for binary conversion before contour detection
        
    Returns:
        Dict: Image with contours and contour information
    """
    try:
        # Read image from path
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        
        # Make a copy for drawing
        img_copy = img.copy()
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply threshold to create binary image
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Map contour modes to OpenCV constants
        contour_modes = {
            "external": cv2.RETR_EXTERNAL,
            "list": cv2.RETR_LIST,
            "ccomp": cv2.RETR_CCOMP,
            "tree": cv2.RETR_TREE
        }
        
        # Map contour methods to OpenCV constants
        contour_methods = {
            "none": cv2.CHAIN_APPROX_NONE,
            "simple": cv2.CHAIN_APPROX_SIMPLE,
            "tc89_l1": cv2.CHAIN_APPROX_TC89_L1,
            "tc89_kcos": cv2.CHAIN_APPROX_TC89_KCOS
        }
        
        # Validate mode and method
        if mode.lower() not in contour_modes:
            raise ValueError(f"Unsupported contour mode: {mode}")
        if method.lower() not in contour_methods:
            raise ValueError(f"Unsupported contour method: {method}")
        
        # Find contours
        contour_mode = contour_modes[mode.lower()]
        contour_method = contour_methods[method.lower()]
        contours, hierarchy = cv2.findContours(binary, contour_mode, contour_method)
        
        # Collect contour data
        contour_data = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Extract simplified contour points
            # Convert to list of [x, y] points for JSON serialization
            points = contour.reshape(-1, 2).tolist()
            
            contour_data.append({
                "index": i,
                "area": float(area),
                "perimeter": float(perimeter),
                "point_count": len(points),
                "bounding_rect": [int(val) for val in cv2.boundingRect(contour)],
                "center": [
                    int(np.mean([p[0] for p in points])), 
                    int(np.mean([p[1] for p in points]))
                ]
                # Not including all points as it could make the response very large
            })
        
        # Draw contours if requested
        if draw:
            cv2.drawContours(img_copy, contours, -1, color, thickness)
        
        # Save result image
        result_path = save_and_display(img_copy, image_path, f"contours_{mode}_{method}")
        
        # Save binary image for reference
        binary_path = save_and_display(binary, image_path, f"binary_for_contours")
        
        return {
            "contour_count": len(contours),
            "contour_info": contour_data[:10],  # Limiting to first 10 to avoid excessive response size
            "contour_parameters": {
                "mode": mode,
                "method": method,
                "threshold_value": threshold_value
            },
            "binary_path": binary_path,
            "path": result_path,
            "output_path": result_path,  # Return path for chaining operations
            "info": get_image_info(img_copy)
        }
        
    except Exception as e:
        logger.error(f"Error detecting contours: {str(e)}")
        raise ValueError(f"Failed to detect contours: {str(e)}")

def find_shapes_tool(
    image_path: str, 
    shape_type: str, 
    param1: float = 100.0, 
    param2: float = 30.0,
    min_radius: int = 0,
    max_radius: int = 0,
    min_dist: int = 50,
    threshold: float = 150.0,
    min_line_length: float = 50.0,
    max_line_gap: float = 10.0,
    draw: bool = True,
    thickness: int = 2,
    color: Tuple[int, int, int] = (0, 0, 255)
) -> Dict[str, Any]:
    """
    Find basic shapes in an image
    
    Args:
        image_path: Path to the image file
        shape_type: Type of shape to find ('circles', 'lines', 'lines_p')
        param1: First method-specific parameter (depends on shape_type)
        param2: Second method-specific parameter (depends on shape_type)
        min_radius: Minimum circle radius (for circles)
        max_radius: Maximum circle radius (for circles)
        min_dist: Minimum distance between detected circles
        threshold: Threshold for Hough transform (for lines)
        min_line_length: Minimum line length (for probabilistic Hough transform)
        max_line_gap: Maximum gap between line segments (for probabilistic Hough transform)
        draw: Whether to draw the detected shapes
        thickness: Thickness of the drawn shapes
        color: Color for drawing shapes (BGR format)
        
    Returns:
        Dict: Image with shapes and shape information
    """
    try:
        # Read image from path
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        
        # Make a copy for drawing
        img_copy = img.copy()
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        shape_info = {"shape_type": shape_type}
        shapes_data = []
        
        if shape_type.lower() == 'circles':
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=min_dist,
                param1=param1,
                param2=param2,
                minRadius=min_radius,
                maxRadius=max_radius if max_radius > 0 else None
            )
            
            shape_info.update({
                "min_dist": min_dist,
                "param1": param1,
                "param2": param2,
                "min_radius": min_radius,
                "max_radius": max_radius
            })
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                
                for i, circle in enumerate(circles[0, :]):
                    center = (circle[0], circle[1])
                    radius = circle[2]
                    
                    shapes_data.append({
                        "index": i,
                        "center": (int(center[0]), int(center[1])),
                        "radius": int(radius),
                        "diameter": int(radius * 2),
                        "area": float(np.pi * radius * radius)
                    })
                    
                    # Draw the circle
                    if draw:
                        cv2.circle(img_copy, center, radius, color, thickness)
                        # Draw the center point
                        cv2.circle(img_copy, center, 2, (0, 255, 0), 3)
            
        elif shape_type.lower() == 'lines':
            # Standard Hough Transform
            lines = cv2.HoughLines(
                blurred,
                rho=1,
                theta=np.pi/180,
                threshold=int(threshold)
            )
            
            shape_info.update({
                "threshold": threshold
            })
            
            if lines is not None:
                for i, line in enumerate(lines):
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    
                    # Calculate line endpoints for a reasonable length
                    length = max(img.shape[0], img.shape[1])
                    x1 = int(x0 + length * (-b))
                    y1 = int(y0 + length * (a))
                    x2 = int(x0 - length * (-b))
                    y2 = int(y0 - length * (a))
                    
                    shapes_data.append({
                        "index": i,
                        "rho": float(rho),
                        "theta": float(theta),
                        "angle_degrees": float(theta * 180 / np.pi)
                    })
                    
                    # Draw the line
                    if draw:
                        cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
            
        elif shape_type.lower() == 'lines_p':
            # Probabilistic Hough Transform
            lines_p = cv2.HoughLinesP(
                blurred,
                rho=1,
                theta=np.pi/180,
                threshold=int(threshold),
                minLineLength=min_line_length,
                maxLineGap=max_line_gap
            )
            
            shape_info.update({
                "threshold": threshold,
                "min_line_length": min_line_length,
                "max_line_gap": max_line_gap
            })
            
            if lines_p is not None:
                for i, line in enumerate(lines_p):
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line length
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    # Calculate angle
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    
                    shapes_data.append({
                        "index": i,
                        "start_point": (int(x1), int(y1)),
                        "end_point": (int(x2), int(y2)),
                        "length": float(length),
                        "angle_degrees": float(angle)
                    })
                    
                    # Draw the line
                    if draw:
                        cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
            
        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")
        
        # Save the result
        result_path = save_and_display(img_copy, image_path, f"shapes_{shape_type}")
        
        return {
            "shape_count": len(shapes_data),
            "shapes": shapes_data,
            "shape_parameters": shape_info,
            "path": result_path,
            "output_path": result_path,  # Return path for chaining operations
            "info": get_image_info(img_copy)
        }
        
    except Exception as e:
        logger.error(f"Error finding shapes: {str(e)}")
        raise ValueError(f"Failed to find shapes: {str(e)}")

def match_template_tool(
    image_path: str, 
    template_path: str, 
    method: str = "ccoeff_normed",
    threshold: float = 0.8,
    draw: bool = True,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> Dict[str, Any]:
    """
    Find a template in an image
    
    Args:
        image_path: Path to the source image file
        template_path: Path to the template image file
        method: Template matching method
        threshold: Threshold for good matches
        draw: Whether to draw rectangle around matches
        color: Color for drawing rectangles (BGR format)
        thickness: Thickness of the rectangle
        
    Returns:
        Dict: Image with matches and match information
    """
    try:
        # Read images from paths
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
            
        template = cv2.imread(template_path)
        if template is None:
            raise ValueError(f"Failed to read template from path: {template_path}")
        
        # Make a copy for drawing
        img_copy = img.copy()
        
        # Get template dimensions
        h, w = template.shape[:2]
        
        # Map method names to OpenCV constants
        methods = {
            "sqdiff": cv2.TM_SQDIFF,
            "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
            "ccorr": cv2.TM_CCORR,
            "ccorr_normed": cv2.TM_CCORR_NORMED,
            "ccoeff": cv2.TM_CCOEFF,
            "ccoeff_normed": cv2.TM_CCOEFF_NORMED
        }
        
        # Validate method
        if method.lower() not in methods:
            raise ValueError(f"Unsupported template matching method: {method}")
        
        match_method = methods[method.lower()]
        
        # Apply template matching
        result = cv2.matchTemplate(img, template, match_method)
        
        # For SQDIFF methods, the best matches are at minima
        if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            locations = np.where(result <= (1.0 - threshold))
            sort_order = np.argsort(result[locations])
            is_min = True
            # Find global min
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            best_loc = min_loc
            best_val = min_val
        else:
            locations = np.where(result >= threshold)
            sort_order = np.argsort(-result[locations])  # Negative for descending order
            is_min = False
            # Find global max
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            best_loc = max_loc
            best_val = max_val
        
        # Prepare match data
        match_data = []
        
        # Add best global match
        pt1 = best_loc
        pt2 = (pt1[0] + w, pt1[1] + h)
        
        match_data.append({
            "index": 0,
            "is_global_best": True,
            "value": float(best_val),
            "top_left": (int(pt1[0]), int(pt1[1])),
            "bottom_right": (int(pt2[0]), int(pt2[1])),
            "match_confidence": float(best_val if not is_min else 1.0 - best_val)
        })
        
        # Draw the best match
        if draw:
            cv2.rectangle(img_copy, pt1, pt2, color, thickness)
        
        # Process other matches
        if len(locations) > 0 and len(locations[0]) > 0:
            loc_y, loc_x = locations
            
            # Limit the number of other matches to show
            max_matches = 9  # Max 10 matches including the best one
            other_matches = min(len(loc_y), max_matches)
            
            for i in range(other_matches):
                idx = sort_order[i]
                y, x = loc_y[idx], loc_x[idx]
                
                # Skip if too close to existing matches
                too_close = False
                for match in match_data:
                    mx, my = match["top_left"]
                    if abs(x - mx) < w/2 and abs(y - my) < h/2:
                        too_close = True
                        break
                
                if too_close:
                    continue
                
                pt1 = (x, y)
                pt2 = (x + w, y + h)
                
                value = result[y, x]
                match_data.append({
                    "index": len(match_data),
                    "is_global_best": False,
                    "value": float(value),
                    "top_left": (int(pt1[0]), int(pt1[1])),
                    "bottom_right": (int(pt2[0]), int(pt2[1])),
                    "match_confidence": float(value if not is_min else 1.0 - value)
                })
                
                # Draw this match
                if draw:
                    cv2.rectangle(img_copy, pt1, pt2, color, thickness)
                
                # Limit to max_matches
                if len(match_data) >= 10:
                    break
        
        # Visualize result matrix
        result_norm = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Save results
        matches_path = save_and_display(img_copy, image_path, f"template_matches_{method}")
        visualization_path = save_and_display(result_norm, image_path, f"template_heatmap_{method}")
        
        return {
            "match_count": len(match_data),
            "matches": match_data,
            "match_parameters": {
                "method": method,
                "threshold": threshold,
                "is_min_method": is_min
            },
            "visualization_path": visualization_path,
            "path": matches_path,
            "output_path": matches_path,  # Return path for chaining operations
            "info": get_image_info(img_copy),
            "template_info": get_image_info(template)
        }
        
    except Exception as e:
        logger.error(f"Error matching template: {str(e)}")
        raise ValueError(f"Failed to match template: {str(e)}")

def register_tools(mcp):
    """
    Register all image processing tools with the MCP server
    
    Args:
        mcp: The MCP server instance
    """
    # Register tool implementations
    mcp.add_tool(apply_filter_tool)
    mcp.add_tool(detect_edges_tool)
    mcp.add_tool(apply_threshold_tool)
    mcp.add_tool(detect_contours_tool)
    mcp.add_tool(find_shapes_tool)
    mcp.add_tool(match_template_tool)