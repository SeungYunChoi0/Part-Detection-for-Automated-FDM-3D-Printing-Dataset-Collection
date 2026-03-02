#!/usr/bin/env python3
"""
Pipeline Visualization for Blue Filament
4-Stage Visualization: Input → HSV Mask → Spatial Filtering → Output

Key Feature: Prime tower exclusion (upper-left 25%×35%)

Author: Pipeline Visualization Script
Date: 2026-02-26
"""

import cv2
import numpy as np
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HSV Parameters for Blue
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOWER_BLUE = np.array([90, 50, 50])
UPPER_BLUE = np.array([130, 255, 255])

# Filtering Parameters
MIN_AREA = 3000
MAX_AREA = 200000
MIN_ASPECT_RATIO = 1.0
MAX_ASPECT_RATIO = 6.0
MIN_EXTENT = 0.25
CENTER_REGION = 0.8

# Blue-specific: Prime Tower Exclusion
PRIME_TOWER_X_MAX = 0.25  # Left 25%
PRIME_TOWER_Y_MAX = 0.35  # Top 35%

# Morphology
KERNEL_SIZE = 7

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def count_holes(contour_idx, hierarchy):
    """Count number of holes (children) in a contour."""
    if hierarchy is None or contour_idx >= len(hierarchy[0]):
        return 0
    
    num_holes = 0
    child_idx = hierarchy[0][contour_idx][2]
    
    while child_idx != -1:
        num_holes += 1
        if child_idx < len(hierarchy[0]):
            child_idx = hierarchy[0][child_idx][0]
        else:
            break
    
    return num_holes


def enhance_image(img):
    """Apply enhancement: LAB + CLAHE + Gamma + Denoising + Sharpening."""
    # LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)
    
    # Gamma correction
    gamma = 0.9
    l_gamma = np.array(255 * (l_eq / 255) ** gamma, dtype=np.uint8)
    
    # Merge and convert back to BGR
    lab_final = cv2.merge([l_gamma, a, b])
    result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
    
    # Denoising
    result = cv2.fastNlMeansDenoisingColored(
        result, None, h=6, hColor=6, 
        templateWindowSize=7, searchWindowSize=21
    )
    
    # Sharpening
    kernel_sharp = np.array([[ 0, -1,  0],
                             [-1,  5, -1],
                             [ 0, -1,  0]])
    result = cv2.filter2D(result, -1, kernel_sharp)
    
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Pipeline with Visualization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def visualize_pipeline(input_path, output_dir):
    """
    Visualize 4-stage pipeline and save intermediate results.
    
    Args:
        input_path: Path to input image
        output_dir: Directory to save visualization results
    
    Returns:
        dict: Paths to saved images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem = Path(input_path).stem
    
    print(f"\n{'='*60}")
    print(f"Processing: {stem}")
    print(f"{'='*60}\n")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (a) STAGE 1: Input - Original Image
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    img = cv2.imread(input_path)
    if img is None:
        print(f"[ERROR] Failed to load image: {input_path}")
        return None
    
    height, width = img.shape[:2]
    
    # Save (a) Input
    input_save_path = output_dir / f"{stem}_a_input.jpg"
    cv2.imwrite(str(input_save_path), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    print(f"[SAVE] (a) Input: {input_save_path}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (b) STAGE 2: HSV Mask
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HSV conversion
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Color masking
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    
    # Morphology
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Save (b) HSV Mask
    mask_save_path = output_dir / f"{stem}_b_hsv_mask.jpg"
    cv2.imwrite(str(mask_save_path), mask, [cv2.IMWRITE_JPEG_QUALITY, 98])
    print(f"[SAVE] (b) HSV Mask: {mask_save_path}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (c) STAGE 3: Spatial Filtering Visualization
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Contour detection
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    print(f"Detected contours: {len(contours)}")
    
    # Create visualization image
    img_bbox = img.copy()
    
    valid_objects = []
    rejected_objects = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Area filtering
        if not (MIN_AREA < area < MAX_AREA):
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / float(min(w, h))
        
        # Aspect ratio filtering
        if not (MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO):
            continue
        
        # Extent filtering
        rect_area = w * h
        extent = area / float(rect_area)
        if extent < MIN_EXTENT:
            continue
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Blue-specific: Prime Tower Exclusion
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        in_prime_tower = (x < width * PRIME_TOWER_X_MAX and 
                          y < height * PRIME_TOWER_Y_MAX)
        
        if in_prime_tower:
            rejected_objects.append({
                'bbox': (x, y, w, h),
                'reason': 'Prime Tower Region'
            })
            continue
        
        # Center region check
        center_x = x + w/2
        center_y = y + h/2
        margin = (1 - CENTER_REGION) / 2
        in_center = (width*margin < center_x < width*(1-margin) and
                     height*margin < center_y < height*(1-margin))
        
        # Count holes
        num_holes = count_holes(i, hierarchy)
        
        # Hole density check
        hole_density = num_holes / (area / 1000)
        if hole_density > 6.0:
            rejected_objects.append({
                'bbox': (x, y, w, h),
                'reason': 'High hole density'
            })
            continue
        
        valid_objects.append({
            'bbox': (x, y, w, h),
            'area': area,
            'holes': num_holes,
            'in_center': in_center
        })
    
    print(f"Valid objects: {len(valid_objects)}")
    print(f"Rejected objects: {len(rejected_objects)}")
    
    # Draw rejected objects (red, dashed)
    for obj in rejected_objects:
        x, y, w, h = obj['bbox']
        dash_length = 10
        color = (0, 0, 255)  # Red
        thickness = 2
        
        # Draw dashed rectangle
        for i in range(0, w, dash_length*2):
            cv2.line(img_bbox, (x+i, y), (x+min(i+dash_length, w), y), color, thickness)
        for i in range(0, w, dash_length*2):
            cv2.line(img_bbox, (x+i, y+h), (x+min(i+dash_length, w), y+h), color, thickness)
        for i in range(0, h, dash_length*2):
            cv2.line(img_bbox, (x, y+i), (x, y+min(i+dash_length, h)), color, thickness)
        for i in range(0, h, dash_length*2):
            cv2.line(img_bbox, (x+w, y+i), (x+w, y+min(i+dash_length, h)), color, thickness)
        
        # Add reason label
        label = obj['reason']
        cv2.putText(img_bbox, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Priority selection
    selected_obj = None
    if len(valid_objects) > 0:
        # Center objects first
        center_objects = [obj for obj in valid_objects if obj['in_center']]
        if len(center_objects) == 0:
            center_objects = valid_objects
        
        # Sort by (holes > 0, area)
        center_objects.sort(key=lambda x: (x['holes'] > 0, x['area']), reverse=True)
        selected_obj = center_objects[0]
        
        # Draw all valid objects (blue)
        for obj in valid_objects:
            if obj == selected_obj:
                continue
            x, y, w, h = obj['bbox']
            cv2.rectangle(img_bbox, (x, y), (x+w, y+h), (255, 255, 0), 2)  # Cyan
        
        # Draw selected object (green, thick)
        x, y, w, h = selected_obj['bbox']
        cv2.rectangle(img_bbox, (x, y), (x+w, y+h), (0, 255, 0), 4)  # Green
        
        # Add label
        label = f"Selected (holes={selected_obj['holes']}, area={selected_obj['area']:.0f})"
        cv2.putText(img_bbox, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        print(f"\n[SELECTED] Bbox: ({x}, {y}, {w}, {h})")
        print(f"           Area: {selected_obj['area']:.0f}px²")
        print(f"           Holes: {selected_obj['holes']}")
        print(f"           Center: {selected_obj['in_center']}")
    
    # Save (c) Spatial Filtering
    bbox_save_path = output_dir / f"{stem}_c_spatial_filtering.jpg"
    cv2.imwrite(str(bbox_save_path), img_bbox, [cv2.IMWRITE_JPEG_QUALITY, 98])
    print(f"\n[SAVE] (c) Spatial Filtering: {bbox_save_path}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (d) STAGE 4: Output - Cropped & Enhanced
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if selected_obj is not None:
        x, y, w, h = selected_obj['bbox']
        
        # Add 15% margin
        margin = int(min(w, h) * 0.15)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(width, x + w + margin)
        y2 = min(height, y + h + margin)
        
        # Crop
        cropped = img[y1:y2, x1:x2].copy()
        
        # Enhancement
        enhanced = enhance_image(cropped)
        
        # Save (d) Output
        output_save_path = output_dir / f"{stem}_d_output.jpg"
        cv2.imwrite(str(output_save_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 98])
        print(f"[SAVE] (d) Output: {output_save_path}")
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS: 4-stage visualization completed")
        print(f"{'='*60}\n")
        
        return {
            'input': str(input_save_path),
            'mask': str(mask_save_path),
            'bbox': str(bbox_save_path),
            'output': str(output_save_path)
        }
    else:
        print(f"\n{'='*60}")
        print(f"❌ FAILED: No valid object detected")
        print(f"{'='*60}\n")
        
        return {
            'input': str(input_save_path),
            'mask': str(mask_save_path),
            'bbox': str(bbox_save_path),
            'output': None
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Batch Processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_combined_visualization(output_dir, stem):
    """
    Create a single combined image with all 4 stages side-by-side.
    """
    output_dir = Path(output_dir)
    
    # Load all 4 images
    img_a = cv2.imread(str(output_dir / f"{stem}_a_input.jpg"))
    img_b = cv2.imread(str(output_dir / f"{stem}_b_hsv_mask.jpg"))
    img_c = cv2.imread(str(output_dir / f"{stem}_c_spatial_filtering.jpg"))
    img_d_path = output_dir / f"{stem}_d_output.jpg"
    
    if not img_d_path.exists():
        print(f"[WARNING] Output image not found, skipping combined visualization")
        return None
    
    img_d = cv2.imread(str(img_d_path))
    
    # Convert mask to BGR for consistency
    if len(img_b.shape) == 2:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)
    
    # Resize all to same height
    target_height = 400
    
    def resize_keep_aspect(img, target_h):
        h, w = img.shape[:2]
        aspect = w / h
        target_w = int(target_h * aspect)
        return cv2.resize(img, (target_w, target_h))
    
    img_a_resized = resize_keep_aspect(img_a, target_height)
    img_b_resized = resize_keep_aspect(img_b, target_height)
    img_c_resized = resize_keep_aspect(img_c, target_height)
    img_d_resized = resize_keep_aspect(img_d, target_height)
    
    # Add labels
    def add_label(img, text):
        img_labeled = img.copy()
        cv2.rectangle(img_labeled, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img_labeled, text, (10, 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img_labeled
    
    img_a_labeled = add_label(img_a_resized, "(a) Input")
    img_b_labeled = add_label(img_b_resized, "(b) HSV Mask")
    img_c_labeled = add_label(img_c_resized, "(c) Spatial Filtering")
    img_d_labeled = add_label(img_d_resized, "(d) Output")
    
    # Concatenate horizontally
    combined = np.hstack([img_a_labeled, img_b_labeled, img_c_labeled, img_d_labeled])
    
    # Save combined
    combined_path = output_dir / f"{stem}_combined.jpg"
    cv2.imwrite(str(combined_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    print(f"[SAVE] Combined visualization: {combined_path}")
    
    return str(combined_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Execution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_pipeline_blue.py <input_image> [output_dir]")
        print("\nExample:")
        print("  python visualize_pipeline_blue.py input.jpg visualization_blue/")
        print("\nBlue-specific features:")
        print("  - Prime tower exclusion (upper-left 25%×35%)")
        print("  - Low V-channel threshold (50) for dark blue")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./visualization_blue"
    
    # Run visualization
    results = visualize_pipeline(input_path, output_dir)
    
    if results and results['output'] is not None:
        # Create combined visualization
        stem = Path(input_path).stem
        create_combined_visualization(output_dir, stem)
        
        print("\n" + "="*60)
        print("📊 Visualization Results:")
        print("="*60)
        for stage, path in results.items():
            print(f"  {stage:15s}: {path}")
        print("="*60 + "\n")
