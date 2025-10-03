#utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.4):
    """Overlay segmentation mask on image"""
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    mask_colored = np.zeros_like(img_array)
    mask_colored[mask > 0] = color
    
    overlayed = cv2.addWeighted(img_array, 1, mask_colored, alpha, 0)
    return Image.fromarray(overlayed)

def draw_measurements(image, measurements):
    """Draw measurement lines and annotations"""
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Draw cardiothoracic ratio lines if available
    if 'heart_width' in measurements and 'thorax_width' in measurements:
        h, w = img_array.shape[:2]
        heart_w = measurements['heart_width']
        thorax_w = measurements['thorax_width']
        
        # Draw horizontal lines
        y_center = h // 2
        cv2.line(img_array, (w//2 - heart_w//2, y_center), 
                (w//2 + heart_w//2, y_center), (0, 255, 0), 2)
        cv2.line(img_array, (w//2 - thorax_w//2, y_center + 50), 
                (w//2 + thorax_w//2, y_center + 50), (255, 0, 0), 2)
    
    return Image.fromarray(img_array)

def draw_rib_markers(image, rib_positions):
    """Mark detected ribs"""
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    for i, pos in enumerate(rib_positions):
        cv2.circle(img_array, (int(pos[0]), int(pos[1])), 5, (0, 255, 255), -1)
        cv2.putText(img_array, f'R{i+1}', (int(pos[0])+10, int(pos[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return Image.fromarray(img_array)

def create_comparison_figure(original, segmented, measurements_img):
    """Create comparison figure with all visualizations"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original X-Ray')
    axes[0].axis('off')
    
    axes[1].imshow(segmented)
    axes[1].set_title('Segmentation Overlay')
    axes[1].axis('off')
    
    axes[2].imshow(measurements_img)
    axes[2].set_title('Measurements')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig