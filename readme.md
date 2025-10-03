# AI Chest X-Ray Analyzer

Comprehensive automated chest X-ray analysis system with structured radiology reporting powered by multiple ML models and Groq's Vision LLM.

## Features

### 1. Technical Considerations
- **Projection Classification**: AP vs PA detection (PA preferred for accuracy)
- **Rotation Assessment**: Clavicle symmetry analysis
- **Inspiration Adequacy**: Automated posterior rib counting (≥9 ribs optimal)
- **Penetration Quality**: Vertebral body visibility check

### 2. Airway and Mediastinum
- Trachea midline position detection
- Mediastinal width assessment
- Carina and bronchi evaluation

### 3. Lungs and Pleura
- Automated lung field segmentation
- Multi-label abnormality detection (consolidation, effusion, pneumothorax, etc.)
- Pleural space assessment
- Costophrenic angle sharpness evaluation

### 4. Cardiac and Great Vessels
- Automated cardiothoracic ratio (CTR) calculation
- Heart size classification (normal: CTR < 0.50)
- Cardiac silhouette assessment

### 5. Bones and Soft Tissue
- Rib and clavicle integrity checks
- Spine assessment
- Soft tissue evaluation

### 6. AI-Powered Report Generation
- Structured radiology report using Groq Vision LLM
- Comprehensive findings summary
- Clinical correlation guidance

## Architecture & Models Used

### Pre-trained Models
1. **ianpan/chest-x-ray-basic** (Primary Model) - Multi-task model providing:
   - View classification (AP/PA/Lateral) - 99.42% accuracy
   - Anatomical segmentation (right lung, left lung, heart) - Dice 0.943-0.957
   - Patient age prediction (MAE: 5.25 years)
   - Patient sex prediction (0.999 AUC)
2. **torchxrayvision** - Multi-label abnormality classification (14+ pathologies)
3. **Groq Vision LLM** (llama-3.2-90b-vision-preview) - Comprehensive report generation

### Segmentation Performance
The ianpan model provides high-quality segmentation with:
- Right Lung: 0.957 Dice coefficient
- Left Lung: 0.948 Dice coefficient  
- Heart: 0.943 Dice coefficient

Automatic fallback to traditional morphological segmentation if ML model unavailable.

### Custom Algorithms
- Morphological lung/heart segmentation with contour analysis
- Automated rib detection using horizontal profile analysis
- Cardiothoracic ratio calculation from segmented masks
- Clavicle symmetry analysis for rotation detection
- Costophrenic angle sharpness evaluation

## Installation

### 1. Clone Repository
```bash
git clone <your-repo>
cd chest-xray-analyzer
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here  # Optional
```

**Get API Keys:**
- Groq API: https://console.groq.com/
- Hugging Face: https://huggingface.co/settings/tokens

## Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the App
1. Upload a chest X-ray image (PNG, JPG, or JPEG)
2. Click "🔍 Analyze X-Ray"
3. Wait for analysis to complete (30-60 seconds)
4. Review results across multiple tabs:
   - **Report**: Full structured radiology report
   - **Technical**: Quality assessment details
   - **Anatomy**: Anatomical findings
   - **Findings**: Detected abnormalities
   - **Visualizations**: Segmentation overlays and measurements

### Download Report
Click "Download Report" button to save the structured report as a text file.

## File Structure

```
chest-xray-analyzer/
├── .env                          # API keys (create from .env.example)
├── .env.example                  # Template for environment variables
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── app.py                       # Main Streamlit application
├── models/
│   ├── __init__.py
│   ├── segmentation.py          # Lung/heart/mediastinum segmentation
│   ├── classification.py        # View classification & abnormality detection
│   ├── measurements.py          # CTR, rib counting, angle assessment
│   └── report_generator.py     # Groq LLM integration for reports
└── utils/
    ├── __init__.py
    ├── preprocessing.py         # Image preprocessing utilities
    └── visualization.py         # Overlay masks and annotations
```

## Pipeline Overview

```
Input X-Ray Image
       ↓
1. Preprocessing & Enhancement (CLAHE)
       ↓
2. Anatomy Segmentation
   - Lungs (bilateral)
   - Heart
   - Mediastinum
       ↓
3. Technical Quality Assessment
   - View classification (AP/PA)
   - Rotation check (clavicle symmetry)
   - Inspiration (rib counting)
   - Penetration (exposure quality)
       ↓
4. Abnormality Detection
   - Multi-label classification
   - 14+ pathology types
       ↓
5. Measurements & Calculations
   - Cardiothoracic ratio
   - Trachea position
   - Costophrenic angles
       ↓
6. Report Generation (Groq Vision LLM)
   - Structured radiology report
   - Clinical interpretation
       ↓
Output: Comprehensive Analysis + Report
```

## Technical Details

### Image Preprocessing
- Contrast enhancement using CLAHE
- Normalization for different model inputs
- Resize to model-specific dimensions (224x224, 512x512)

### Segmentation Approach
- Otsu thresholding for lung fields
- Morphological operations (opening, closing)
- Contour detection and filtering
- Region-based heart localization

### Measurement Algorithms
- **Rib Counting**: Horizontal intensity profile analysis with peak detection
- **CTR Calculation**: Bounding box width ratio (heart/thorax)
- **Rotation Check**: Left-right clavicle region similarity
- **Angle Assessment**: Convexity defect analysis at costophrenic angles

### Abnormality Detection
Uses torchxrayvision DenseNet-121 trained on multiple datasets:
- Atelectasis
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural Thickening
- Cardiomegaly
- Nodule
- Mass
- Hernia

## Limitations & Disclaimers

⚠️ **IMPORTANT MEDICAL DISCLAIMER**

This application is an AI-assisted educational and research tool. It is NOT intended for:
- Clinical diagnosis
- Treatment decisions
- Replacing qualified radiologists
- Emergency medical use

**All findings must be verified by a licensed radiologist before any clinical action.**

### Technical Limitations
- Segmentation accuracy depends on image quality
- Abnormality detection has inherent false positive/negative rates
- CTR calculation requires proper PA view and good image quality
- Rib counting may be affected by patient positioning
- Report generation quality depends on Groq API availability

### Best Practices
- Use high-quality, properly positioned X-rays
- Prefer PA view over AP for accurate CTR
- Ensure adequate inspiration (patient took deep breath)
- Verify technical parameters before clinical interpretation
- Always correlate with clinical findings

## Performance

### Speed (on CPU)
- Image loading: < 1s
- Segmentation: 2-5s
- Classification: 3-8s
- Measurements: 1-2s
- Report generation: 5-10s
- **Total: 30-60 seconds per image**

### Speed (on GPU)
- Classification: 1-2s
- **Total: 10-20 seconds per image**

## Troubleshooting

### Common Issues

**1. Models not loading**
```bash
# Clear cache and reinstall
pip cache purge
pip install --upgrade --force-reinstall torchxrayvision transformers
```

**2. Groq API errors**
- Check API key in `.env` file
- Verify API quota at https://console.groq.com/
- App falls back to template reports if API unavailable

**3. Memory errors**
- Reduce image size before upload
- Close other applications
- Use smaller batch sizes (already implemented)

**4. CUDA errors**
- Update GPU drivers
- Install compatible PyTorch version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Development

### Adding New Features

**Add new abnormality detector:**
```python
# In models/classification.py
def load_custom_model(self):
    from transformers import AutoModel
    self.custom_model = AutoModel.from_pretrained("your-model")
```

**Add new measurement:**
```python
# In models/measurements.py
def measure_custom_feature(self, image, mask):
    # Your measurement logic
    return result
```

**Customize report template:**
```python
# In models/report_generator.py
def _generate_template_report(self, findings_summary):
    # Modify template structure
    return custom_report
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{chest_xray_analyzer,
  title={AI Chest X-Ray Analyzer},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/chest-xray-analyzer}
}
```

### Models Used
- **torchxrayvision**: Cohen, J. P. et al. (2020)
- **Groq LLM**: Groq Inc. (2024)
- **Hugging Face Models**: Various authors (see model cards)

## License

This project is for educational and research purposes only. 

**Medical Imaging Data**: Ensure compliance with HIPAA, GDPR, and local regulations when handling patient data.

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: your.email@example.com

## Roadmap

- [ ] DICOM file support
- [ ] Batch processing multiple images
- [ ] Comparison with previous studies
- [ ] PDF report generation with images
- [ ] Integration with PACS systems
- [ ] Mobile app version
- [ ] Multi-language report support
- [ ] 3D visualization for CT scans

## Acknowledgments

- TorchXRayVision team for pre-trained models
- Groq for vision LLM API
- Hugging Face community for model hosting
- Open-source medical imaging community

---

**Built with ❤️ for medical AI research and education**