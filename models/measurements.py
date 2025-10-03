#models/measurements.py

import numpy as np
import cv2
from scipy import ndimage

class XRayMeasurements:
    def __init__(self):
        pass
    
    def count_posterior_ribs(self, image, lung_mask):
        """Count visible posterior ribs"""
        # Convert to grayscale
        if len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = image
        
        h, w = img_gray.shape
        
        # Focus on right lung area with wider ROI to capture more ribs while maintaining medial focus
        # Still avoid apices where anterior ribs dominate
        roi_x1, roi_x2 = int(w * 0.30), int(w * 0.70)  # Wider ROI
        roi_y1, roi_y2 = int(h * 0.2), int(h * 0.85)  # Extended lower bound for better rib visibility
        
        roi = img_gray[roi_y1:roi_y2, roi_x1:roi_x2]
        lung_roi = lung_mask[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Dilate lung mask to avoid cutting off ribs at edges
        kernel = np.ones((7,7), np.uint8)
        lung_roi_dilated = cv2.dilate(lung_roi, kernel, iterations=1)
        
        # Apply dilated lung mask
        masked_roi = cv2.bitwise_and(roi, roi, mask=lung_roi_dilated)
        
        # Enhance rib structures with stronger contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(masked_roi)
        
        # Calculate full gradient magnitude but filter for near-horizontal orientations
        sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=5)
        
        # Compute gradient magnitude and angle
        magnitude = np.hypot(sobelx, sobely)
        angle = np.degrees(np.arctan2(sobely, sobelx))
        
        # Filter for near-horizontal edges with more tolerance (60-120 degrees)
        horizontal_mask = (angle > 60) & (angle < 120)  # More permissive angle range
        posterior_edges = (magnitude * horizontal_mask)
        # Add epsilon to prevent division by zero and improve weak edge detection
        posterior_edges = (posterior_edges / (posterior_edges.max() + 1e-6) * 255).astype(np.uint8)
        
        # Profile across rows (posterior ribs show as horizontal peaks)
        horizontal_profile = np.sum(posterior_edges, axis=1)
        
        # Smooth profile with wider kernel for better noise reduction
        from scipy.signal import find_peaks
        smoothed = np.convolve(horizontal_profile, np.ones(15)/15, mode='same')
        
        # Use adaptive distance based on image height for peak detection
        min_dist = max(10, h // 25)  # Approximate rib spacing
        peaks, properties = find_peaks(smoothed, distance=min_dist, prominence=150)  # Lower prominence threshold
        
        # Sort peaks by prominence to keep strongest ones
        peak_prominences = properties['prominences']
        sorted_indices = np.argsort(peak_prominences)[::-1]
        peaks = peaks[sorted_indices]
        
        rib_count = len(peaks)
        
        # Estimate rib positions
        rib_positions = []
        for peak in peaks:
            y_pos = roi_y1 + peak
            x_pos = roi_x1 + int((roi_x2 - roi_x1) / 2)
            rib_positions.append((x_pos, y_pos))
        
        # Assessment
        if rib_count >= 8:
            assessment = "Adequate inspiration (≥8 posterior ribs visible)"
        elif rib_count >= 6:
            assessment = "Borderline inspiration (6-7 posterior ribs visible)"
        else:
            assessment = "Poor inspiration (<6 posterior ribs visible)"
        
        return {
            'count': rib_count,
            'positions': rib_positions,
            'assessment': assessment
        }
    
    def calculate_cardiothoracic_ratio(self, heart_mask, lung_mask):
        """Calculate cardiothoracic ratio (CTR)"""
        # Get bounding boxes
        heart_contours, _ = cv2.findContours(heart_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lung_contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not heart_contours or not lung_contours:
            return None
        
        # Get heart width (maximum horizontal extent)
        heart_contour = max(heart_contours, key=cv2.contourArea)
        heart_x = [pt[0][0] for pt in heart_contour]
        heart_width = max(heart_x) - min(heart_x) if heart_x else 0
        
        # Get thorax width (maximum horizontal extent of both lungs)
        all_lung_points = np.vstack(lung_contours)
        thorax_x = [pt[0][0] for pt in all_lung_points]
        thorax_width = max(thorax_x) - min(thorax_x) if thorax_x else 1
        
        # Calculate CTR
        ctr = heart_width / thorax_width if thorax_width > 0 else 0
        
        # Assessment
        if ctr < 0.5:
            assessment = f"Normal heart size (CTR: {ctr:.2f}, <0.50)"
        elif ctr < 0.55:
            assessment = f"Borderline cardiomegaly (CTR: {ctr:.2f})"
        else:
            assessment = f"Cardiomegaly (CTR: {ctr:.2f}, >0.50)"
        
        return {
            'ctr': ctr,
            'heart_width': heart_width,
            'thorax_width': thorax_width,
            'assessment': assessment
        }
    
    def check_penetration(self, image):
        """Check if vertebral bodies are faintly visible through cardiac shadow"""
        if len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = image
        
        h, w = img_gray.shape
        
        # Focus on mid-thoracic region (cardiac shadow area)
        roi_y1, roi_y2 = int(h * 0.4), int(h * 0.7)
        roi_x1, roi_x2 = int(w * 0.35), int(w * 0.65)
        
        cardiac_roi = img_gray[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Calculate mean intensity
        mean_intensity = np.mean(cardiac_roi)
        std_intensity = np.std(cardiac_roi)
        
        # Check for vertebral visibility (should have some variation/structure)
        # Good penetration: mean around 80-120, std > 20
        if 70 < mean_intensity < 130 and std_intensity > 15:
            assessment = "Appropriate exposure - vertebral bodies faintly visible"
            adequate = True
        elif mean_intensity < 70:
            assessment = "Possible over-penetration - image too dark"
            adequate = False
        else:
            assessment = "Possible under-penetration - image too light"
            adequate = False
        
        return {
            'adequate': adequate,
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity),
            'assessment': assessment
        }
    
    def check_costophrenic_angles(self, lung_mask):
        """Check if costophrenic angles are sharp and well-defined using improved geometric analysis"""
        h, w = lung_mask.shape
        
        # Clean up mask with morphological operations
        kernel = np.ones((5,5), np.uint8)
        cleaned_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # Find lung contours with more points for better analysis
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) < 2:
            return {
                'sharp': False,
                'assessment': 'Unable to evaluate costophrenic angles',
                'confidence': 0.0
            }
        
        # Sort by area and position to get left and right lungs
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        contours = sorted(contours, key=lambda c: c[0][0][0])  # Sort by x position
        
        angles_sharp = True
        angles_data = []
        
        for i, contour in enumerate(contours):
            # Find lowest point (costophrenic angle)
            lowest_point = tuple(contour[contour[:,:,1].argmax()][0])
            
            # Check if angle is well-defined (sharp corner)
            # Get points around lowest point
            idx = contour[:,:,1].argmax()
            window = 20
            start_idx = max(0, idx - window)
            end_idx = min(len(contour), idx + window)
            
            local_contour = contour[start_idx:end_idx]
            
            # Calculate angle sharpness using simpler angle-based method
            if len(local_contour) >= 3:
                # Get points for angle calculation
                mid_point = local_contour[len(local_contour)//2][0]
                start_point = local_contour[0][0]
                end_point = local_contour[-1][0]
                
                # Calculate vectors
                vector1 = start_point - mid_point
                vector2 = end_point - mid_point
                
                # Calculate angle
                cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angle_deg = np.degrees(angle)
                
                # Sharp angle should be acute
                is_sharp = angle_deg < 90  # Consider angles less than 80 degrees as sharp
            else:
                is_sharp = True  # Default to True if not enough points
                angle_deg = None  # No angle could be calculated
            
            angles_data.append({
                'position': lowest_point,
                'sharp': is_sharp,
                'side': 'left' if i == 0 else 'right',
                'angle': float(angle_deg) if angle_deg is not None else None
            })
            
            if not is_sharp:
                angles_sharp = False
        
        # Create detailed assessment with angle measurements
        angle_details = []
        for angle_data in angles_data:
            if angle_data['angle'] is not None:
                angle_details.append(f"{angle_data['side'].capitalize()}: {angle_data['angle']:.1f}°")
        
        if angles_sharp:
            base_assessment = "Costophrenic angles are sharp and well-defined, ruling out pleural effusion"
        else:
            base_assessment = "Costophrenic angles may be blunted - consider pleural effusion"
        
        # Add angle measurements to assessment if available
        if angle_details:
            assessment = f"{base_assessment} (Measured angles: {', '.join(angle_details)})"
        else:
            assessment = base_assessment
        
        return {
            'sharp': angles_sharp,
            'angles': angles_data,
            'assessment': assessment,
            'left_angle': next((data['angle'] for data in angles_data if data['side'] == 'left'), None),
            'right_angle': next((data['angle'] for data in angles_data if data['side'] == 'right'), None)
        }
    
    def _calculate_costophrenic_angle_precise(self, contour, side, img_h, img_w):
        """
        Calculate costophrenic angle using curve fitting and intersection detection
        
        Method:
        1. Identify the costophrenic region (lower lateral aspect)
        2. Fit curves to the diaphragm (bottom) and chest wall (lateral)
        3. Find intersection point
        4. Calculate angle at intersection using tangent vectors
        """
        try:
            # Extract contour points
            points = contour.reshape(-1, 2)
            
            # Find the lowest point as reference
            lowest_idx = np.argmax(points[:, 1])
            lowest_y = points[lowest_idx, 1]
            lowest_x = points[lowest_idx, 0]
            
            # Define region of interest around costophrenic angle
            # For left lung, focus on lower-left; for right lung, lower-right
            roi_y_min = max(0, int(lowest_y - img_h * 0.15))
            roi_y_max = min(img_h, int(lowest_y + img_h * 0.05))
            
            if side == 'left':
                # Left costophrenic angle: left lateral aspect
                roi_x_min = max(0, int(lowest_x - img_w * 0.1))
                roi_x_max = int(lowest_x + img_w * 0.05)
            else:
                # Right costophrenic angle: right lateral aspect
                roi_x_min = int(lowest_x - img_w * 0.05)
                roi_x_max = min(img_w, int(lowest_x + img_w * 0.1))
            
            # Filter points in ROI
            roi_mask = (
                (points[:, 0] >= roi_x_min) & (points[:, 0] <= roi_x_max) &
                (points[:, 1] >= roi_y_min) & (points[:, 1] <= roi_y_max)
            )
            roi_points = points[roi_mask]
            
            if len(roi_points) < 20:
                return None
            
            # Separate into diaphragm (bottom) and chest wall (lateral) segments
            y_coords = roi_points[:, 1]
            x_coords = roi_points[:, 0]
            
            # Sort points by position to separate segments
            if side == 'left':
                # For left: lateral wall is leftmost, diaphragm is bottom
                lateral_wall_points = roi_points[x_coords <= np.percentile(x_coords, 40)]
                diaphragm_points = roi_points[y_coords >= np.percentile(y_coords, 60)]
            else:
                # For right: lateral wall is rightmost, diaphragm is bottom
                lateral_wall_points = roi_points[x_coords >= np.percentile(x_coords, 60)]
                diaphragm_points = roi_points[y_coords >= np.percentile(y_coords, 60)]
            
            if len(lateral_wall_points) < 5 or len(diaphragm_points) < 5:
                # Fallback to simpler method
                return self._calculate_angle_simple(roi_points, lowest_x, lowest_y, side)
            
            # Fit polynomial curves to each segment
            try:
                # Lateral wall: fit x as function of y (vertical line)
                wall_fit = np.polyfit(lateral_wall_points[:, 1], lateral_wall_points[:, 0], deg=2)
                wall_poly = np.poly1d(wall_fit)
                
                # Diaphragm: fit y as function of x (horizontal curve)
                diaph_fit = np.polyfit(diaphragm_points[:, 0], diaphragm_points[:, 1], deg=2)
                diaph_poly = np.poly1d(diaph_fit)
                
                # Find intersection point (approximate)
                intersection_x = lowest_x
                intersection_y = lowest_y
                
                # Calculate tangent vectors at intersection
                # Lateral wall tangent: derivative of x with respect to y
                wall_deriv = np.polyder(wall_poly)
                dx_dy_wall = wall_deriv(intersection_y)
                wall_tangent = np.array([dx_dy_wall, 1.0])  # (dx/dy, 1)
                wall_tangent = wall_tangent / np.linalg.norm(wall_tangent)
                
                # Diaphragm tangent: derivative of y with respect to x
                diaph_deriv = np.polyder(diaph_poly)
                dy_dx_diaph = diaph_deriv(intersection_x)
                diaph_tangent = np.array([1.0, dy_dx_diaph])  # (1, dy/dx)
                diaph_tangent = diaph_tangent / np.linalg.norm(diaph_tangent)
                
                # Calculate angle between tangent vectors
                cos_angle = np.dot(wall_tangent, diaph_tangent)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                
                # Ensure angle is acute (0-180°)
                if angle_deg > 180:
                    angle_deg = 360 - angle_deg
                
                # Calculate confidence based on fit quality
                wall_residuals = lateral_wall_points[:, 0] - wall_poly(lateral_wall_points[:, 1])
                diaph_residuals = diaphragm_points[:, 1] - diaph_poly(diaphragm_points[:, 0])
                
                wall_rmse = np.sqrt(np.mean(wall_residuals**2))
                diaph_rmse = np.sqrt(np.mean(diaph_residuals**2))
                
                # Confidence decreases with RMSE
                confidence = 1.0 / (1.0 + (wall_rmse + diaph_rmse) / 20.0)
                confidence = np.clip(confidence, 0.3, 1.0)
                
                return {
                    'position': (int(intersection_x), int(intersection_y)),
                    'angle': float(angle_deg),
                    'sharp': angle_deg < 90,
                    'side': side,
                    'confidence': float(confidence),
                    'method': 'curve_fitting'
                }
                
            except (np.linalg.LinAlgError, ValueError):
                # If curve fitting fails, use simpler method
                return self._calculate_angle_simple(roi_points, lowest_x, lowest_y, side)
                
        except Exception as e:
            return None
    
    def _calculate_angle_simple(self, points, center_x, center_y, side):
        """Fallback simple angle calculation method"""
        try:
            # Find points to the left/right and below center
            if side == 'left':
                lateral_points = points[points[:, 0] < center_x]
            else:
                lateral_points = points[points[:, 0] > center_x]
            
            bottom_points = points[points[:, 1] > center_y]
            
            if len(lateral_points) < 3 or len(bottom_points) < 3:
                return None
            
            # Get representative points
            if side == 'left':
                lateral_pt = lateral_points[np.argmin(lateral_points[:, 0])]
            else:
                lateral_pt = lateral_points[np.argmax(lateral_points[:, 0])]
            
            bottom_pt = bottom_points[np.argmax(bottom_points[:, 1])]
            center_pt = np.array([center_x, center_y])
            
            # Calculate vectors from center
            vec1 = lateral_pt - center_pt
            vec2 = bottom_pt - center_pt
            
            # Calculate angle
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return {
                'position': (int(center_x), int(center_y)),
                'angle': float(angle_deg),
                'sharp': angle_deg < 90,
                'side': side,
                'confidence': 0.6,  # Lower confidence for simple method
                'method': 'simple'
            }
        except Exception:
            return None
    
    def check_trachea_position(self, image):
        """Check if trachea appears midline"""
        if len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = image
        
        h, w = img_gray.shape
        
        # Focus on upper thorax where trachea is visible
        roi_y1, roi_y2 = int(h * 0.05), int(h * 0.3)
        trachea_roi = img_gray[roi_y1:roi_y2, :]
        
        # Trachea appears as dark vertical line near center
        # Apply vertical edge detection
        sobelx = cv2.Sobel(trachea_roi, cv2.CV_64F, 1, 0, ksize=3)
        
        # Look for strong vertical edges near center
        center_region = sobelx[:, int(w*0.4):int(w*0.6)]
        vertical_profile = np.sum(np.abs(center_region), axis=0)
        
        # Find peak (likely trachea)
        if len(vertical_profile) > 0:
            trachea_x_relative = np.argmax(vertical_profile)
            trachea_x_absolute = int(w*0.4) + trachea_x_relative
            
            # Check if near midline
            center_x = w // 2
            deviation = abs(trachea_x_absolute - center_x)
            deviation_percent = (deviation / w) * 100
            
            is_midline = deviation_percent < 5  # Within 5% of center
            
            assessment = "Trachea appears midline without deviation" if is_midline else f"Trachea deviation detected ({deviation_percent:.1f}% from midline)"
        else:
            is_midline = True
            assessment = "Trachea position normal"
        
        return {
            'midline': is_midline,
            'assessment': assessment
        }