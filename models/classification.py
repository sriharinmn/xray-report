#models/classification.py

import torch
from transformers import AutoModel
import torchxrayvision as xrv
import numpy as np
import cv2

class XRayClassifier:
    def __init__(self, device='cpu'):
        self.device = device
        self.view_classifier = None
        self.abnormality_model = None
        self.torchxrv_model = None
        
    def load_view_classifier(self):
        """Load AP/PA view classifier - uses custom model with segmentation"""
        try:
            model_name = "ianpan/chest-x-ray-basic"
            self.view_classifier = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.view_classifier.to(self.device)
            self.view_classifier.eval()
            # Store mapping for view predictions
            self.view_mapping = {0: 'AP', 1: 'PA', 2: 'LATERAL'}
            print("View classifier (with segmentation) loaded successfully")
        except Exception as e:
            print(f"Error loading view classifier: {e}")
            self.view_classifier = None
    
    def load_abnormality_detector(self):
        """Load abnormality detection model"""
        try:
            # Using torchxrayvision for multi-label classification
            self.torchxrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.torchxrv_model.to(self.device)
            self.torchxrv_model.eval()
            print("Abnormality detector loaded successfully")
        except Exception as e:
            print(f"Error loading abnormality detector: {e}")
            self.torchxrv_model = None
    
    def classify_view(self, image, preprocessed_tensor=None):
        """Classify if AP or PA view.

        preprocessed_tensor is optional. If provided it should be a grayscale numpy array.
        If not provided, image should be a PIL Image and will be converted to grayscale.
        """
        if self.view_classifier is None:
            raise ValueError("View classifier model not loaded. Please call load_view_classifier() first.")

        try:
            import torch
            # Prepare input tensor
            if preprocessed_tensor is None:
                # Convert PIL image to numpy array in grayscale
                img_array = np.array(image.convert('L'))
                # Normalize and resize if needed
                img_array = img_array.astype(np.float32) / 255.0
                if img_array.shape != (224, 224):
                    img_array = cv2.resize(img_array, (224, 224))
                # Add batch and channel dimensions
                x = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
            else:
                # Handle pre-processed tensor
                x = preprocessed_tensor
                if isinstance(x, np.ndarray):
                    if x.ndim == 2:
                        x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
                    elif x.ndim == 3:
                        x = torch.from_numpy(x[0]).float().unsqueeze(0)
                elif isinstance(x, torch.Tensor):
                    if x.dim() == 2:
                        x = x.float().unsqueeze(0).unsqueeze(0)
                    elif x.dim() == 3:
                        x = x.float().unsqueeze(0)
                    
                # Ensure proper normalization
                if x.max() > 1.0:
                    x = x / 255.0
                    
            x = x.to(self.device)

            # Run inference
            with torch.inference_mode():
                outputs = self.view_classifier(x)
                
                # Model outputs a dictionary with 'mask', 'age', 'view', 'female'
                view_probs = torch.nn.functional.softmax(outputs['view'], dim=-1)
                predicted_class = int(view_probs.argmax().item())
                confidence = float(view_probs.max().item())
                masks = outputs['mask'].argmax(1)[0].cpu().numpy()  # Convert to numpy
                age = float(outputs['age'].item())
                is_female = bool(outputs['female'].item() >= 0.5)
                
                gender = "Female" if is_female else "Male"
                view_label = self.view_mapping[predicted_class]
                
                # Get array for rotation check
                if isinstance(x, torch.Tensor):
                    rotation_array = x.squeeze().cpu().numpy()
                else:
                    rotation_array = x.squeeze()
                
                # Ensure proper range for rotation check
                if rotation_array.max() <= 1.0:
                    rotation_array = (rotation_array * 255).astype(np.uint8)
                
                rotation_check = self.check_rotation(rotation_array)
                
                result = {
                    'view': view_label,
                    'confidence': confidence,
                    'mask': masks,
                    'age': age,
                    'gender': gender,
                    'note': 'Model-based classification with demographics',
                    'rotation': rotation_check,
                    'message': rotation_check['message'],
                    'assessment': rotation_check['result']
                }
                return result

        except Exception as e:
            print(f"Error in view classification: {e}")
            print(f"Input type: {type(preprocessed_tensor)}")
            if isinstance(preprocessed_tensor, torch.Tensor):
                print(f"Tensor shape: {preprocessed_tensor.shape}, dtype: {preprocessed_tensor.dtype}")
            elif isinstance(preprocessed_tensor, np.ndarray):
                print(f"Array shape: {preprocessed_tensor.shape}, dtype: {preprocessed_tensor.dtype}")
            raise
    
    def detect_abnormalities(self, preprocessed_tensor):
        """Detect abnormalities using torchxrayvision"""
        if self.torchxrv_model is None:
            raise ValueError("Abnormality model not loaded. Please call load_abnormality_detector() first.")
        
        try:
            # Ensure tensor is in the right format (1, 1, 224, 224)
            if isinstance(preprocessed_tensor, np.ndarray):
                if preprocessed_tensor.ndim == 2:
                    preprocessed_tensor = torch.from_numpy(preprocessed_tensor).float()
                    preprocessed_tensor = preprocessed_tensor.unsqueeze(0).unsqueeze(0)
                elif preprocessed_tensor.ndim == 3:
                    preprocessed_tensor = torch.from_numpy(preprocessed_tensor[0]).float()
                    preprocessed_tensor = preprocessed_tensor.unsqueeze(0)

            with torch.inference_mode():
                # Ensure tensor is on the right device
                if isinstance(preprocessed_tensor, np.ndarray):
                    preprocessed_tensor = torch.from_numpy(preprocessed_tensor)
                preprocessed_tensor = preprocessed_tensor.to(self.device)
                outputs = self.torchxrv_model(preprocessed_tensor)
                predictions = torch.sigmoid(outputs).cpu().numpy()[0]
                
                # Map predictions to pathologies
                pathologies = self.torchxrv_model.pathologies
                findings = {}
                all_scores = {}  # Store all detection scores
                
                for i, pathology in enumerate(pathologies):
                    score = float(predictions[i])
                    findings[pathology] = score
                    # Format score as percentage with 1 decimal place
                    all_scores[pathology] = f"{score * 100:.1f}%"
                
                # Get significant findings
                significant = self.get_significant_findings(findings)
                
                # Sort all scores by value for better display
                sorted_scores = dict(sorted(
                    all_scores.items(),
                    key=lambda x: float(x[1].rstrip('%')),
                    reverse=True
                ))
                
                # Prepare comprehensive result
                result = {
                    'findings': findings,  # Raw scores for processing
                    'all_scores': sorted_scores,  # Formatted scores for display
                    'significant': significant,  # Findings above threshold
                    'assessment': 'Abnormal' if significant else 'Normal',
                    'message': f'Found {len(significant)} significant findings' if significant else 'No significant findings'
                }
                
                return result

        except Exception as e:
            print(f"Error in abnormality detection: {e}")
            raise
    
    def get_significant_findings(self, abnormalities, threshold=0.6):
        """Filter significant findings above threshold and rank by confidence
        
        Pathology-specific thresholds based on clinical significance and prevalence:
        - Critical urgent findings (pneumothorax): 0.65 (highest threshold due to urgency)
        - Acute findings (edema, pneumonia): 0.70 
        - Chronic/structural findings (effusion, consolidation): 0.65
        - Common findings (cardiomegaly): 0.60
        
        Only returns top 3 most confident findings to avoid overcalling.
        """
        pathology_thresholds = {
            'Pneumothorax': 0.65,  # Critical finding - still conservative
            'Edema': 0.70,         # Acute finding
            'Pneumonia': 0.70,     # Acute finding
            'Consolidation': 0.65, # May be chronic
            'Effusion': 0.65,      # May be chronic
            'Cardiomegaly': 0.60   # Chronic structural change
        }
        
        # If abnormalities is a dict with 'findings' key, use that
        if isinstance(abnormalities, dict) and 'findings' in abnormalities:
            findings = abnormalities['findings']
        else:
            findings = abnormalities
            
        # Get findings above their respective thresholds
        significant = {
            k: v for k, v in findings.items() 
            if isinstance(v, (int, float)) and v > pathology_thresholds.get(k, threshold)
        }
        
        # Sort by confidence and limit to top 3 most confident findings
        sorted_findings = dict(
            sorted(significant.items(), key=lambda x: x[1], reverse=True)[:3]
        )
        
        return sorted_findings
    
    def check_rotation(self, image):
        """Check if image has rotation based on clavicle symmetry"""
        # Convert to numpy array if PIL Image
        if hasattr(image, 'convert'):
            img_array = np.array(image.convert('L'))
        elif isinstance(image, torch.Tensor):
            img_array = image.cpu().numpy()
        else:
            img_array = image
            
        # Ensure 2D array
        if img_array.ndim > 2:
            img_array = img_array[0] if img_array.ndim == 3 else img_array[0, 0]
            
        # Ensure proper data range (0-255)
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
            
        h, w = img_array.shape
        
        # Check top region where clavicles are
        clavicle_region = img_array[int(h*0.1):int(h*0.25), :]
        
        # Compare left and right halves
        left_half = clavicle_region[:, :w//2]
        right_half = np.fliplr(clavicle_region[:, w//2:])
        
        # Calculate similarity
        diff = np.abs(left_half.astype(float) - right_half.astype(float)).mean()
        
        # Lower diff means more symmetric
        is_symmetric = diff < 30  # Threshold
        
        return {
            'symmetric': is_symmetric,
            'rotation_score': float(diff),
            'message': 'Minimal or no rotation' if is_symmetric else 'Possible rotation detected',
            'result': 'normal' if is_symmetric else 'rotated'
        }