#utils/preprocessing.py

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

class XRayPreprocessor:
    def __init__(self):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def load_image(self, image_path_or_file):
        """Load image from path or uploaded file (supports DICOM)"""
        try:
            # Try loading as DICOM first if it's a file path
            if isinstance(image_path_or_file, str) and image_path_or_file.endswith('.dcm'):
                return self.load_dicom(image_path_or_file)
            
            # Otherwise load as regular image
            if isinstance(image_path_or_file, str):
                img = Image.open(image_path_or_file).convert('RGB')
            else:
                # Check if uploaded file is DICOM
                if hasattr(image_path_or_file, 'name') and image_path_or_file.name.endswith('.dcm'):
                    return self.load_dicom(image_path_or_file)
                img = Image.open(image_path_or_file).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image: {e}")
            raise
    
    def load_dicom(self, dicom_file):
        """Load DICOM file and convert to PIL Image"""
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            
            # Read DICOM file
            if isinstance(dicom_file, str):
                ds = pydicom.dcmread(dicom_file)
            else:
                ds = pydicom.dcmread(dicom_file)
            
            # Get pixel array
            img_array = ds.pixel_array
            
            # Apply VOI LUT (window/level)
            img_array = apply_voi_lut(img_array, ds)
            
            # Normalize to 0-255
            img_array = img_array - img_array.min()
            img_array = img_array / img_array.max() * 255
            img_array = img_array.astype(np.uint8)
            
            # Convert to PIL Image
            img = Image.fromarray(img_array).convert('RGB')
            return img
            
        except ImportError:
            print("pydicom not installed. Install with: pip install pydicom")
            raise
        except Exception as e:
            print(f"Error loading DICOM: {e}")
            raise
    
    def preprocess_for_classification(self, img, size=224):
        """Preprocess for ViT models"""
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            self.normalize
        ])
        return transform(img).unsqueeze(0)
    
    def preprocess_for_segmentation(self, img, size=512):
        """Preprocess for segmentation models"""
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        return transform(img).unsqueeze(0)
    
    def preprocess_for_torchxrayvision(self, img):
        """Preprocess for torchxrayvision models"""
        # Convert to grayscale
        if isinstance(img, Image.Image):
            img = np.array(img.convert('L'))
        try:
            import torchxrayvision as xrv
            # xrv.datasets.normalize will apply standard normalization used by models
            # Accept both 0-255 images and DICOM HU ranges; pass 255 for uint8-like input
            img_norm = xrv.datasets.normalize(img, 255)
            # Resize to 224x224
            img_resized = cv2.resize(img_norm, (224, 224))
            # Ensure float32 and channel-first
            arr = img_resized.astype('float32')
            arr = np.expand_dims(arr, axis=0)
            return torch.from_numpy(arr).unsqueeze(0)
        except Exception:
            # Fallback: scale uint8 images to [0,1] and return tensor
            img_f = img.astype(np.float32)

            # Heuristic check: if values look like already normalized [-1,1], warn
            vmin, vmax = img_f.min(), img_f.max()
            if vmin >= -1.1 and vmax <= 1.1:
                # Likely already normalized to [-1,1] — convert to [0,1] conservatively
                img_f = (img_f + 1.0) / 2.0
            else:
                # If image looks like 0-255, scale to [0,1]
                if vmax > 2.0:
                    img_f = img_f / 255.0

            img_resized = cv2.resize(img_f, (224, 224))
            arr = np.expand_dims(img_resized.astype('float32'), axis=0)
            return torch.from_numpy(arr).unsqueeze(0)
    
    def enhance_contrast(self, img):
        """Apply CLAHE for contrast enhancement"""
        img_array = np.array(img.convert('L'))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_array)
        return Image.fromarray(enhanced).convert('RGB')
    
    def detect_edges(self, img):
        """Detect edges using Canny"""
        img_array = np.array(img.convert('L'))
        edges = cv2.Canny(img_array, 50, 150)
        return edges