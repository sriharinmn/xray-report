import streamlit as st
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from groq import Groq
from datetime import datetime
import json
import torchxrayvision as xrv
import skimage
from dotenv import load_dotenv
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="Chest X-Ray Analysis",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .report-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1edff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #0d6efd;
        margin: 1rem 0;
    }
    .finding-box {
        background-color: #f8f9fa;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 3px solid #28a745;
    }
    .abnormal-finding {
        border-left-color: #dc3545 !important;
        background-color: #ffeaea;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_xray_models():
    """Load TorchXRayVision models"""
    try:
        # Load multiple models for comprehensive analysis
        models = {}
        
        # DenseNet model trained on multiple datasets
        models['densenet'] = xrv.models.DenseNet(weights="densenet121-res224-all")
        
        # ResNet model
        models['resnet'] = xrv.models.ResNet(weights="resnet50-res512-all")
        
        # Set models to evaluation mode
        for model in models.values():
            model.eval()
            
        return models
    except Exception as e:
        st.error(f"Error loading XRayVision models: {str(e)}")
        return None

def init_groq_client():
    """Initialize Groq client"""
    try:
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            st.error("GROQ_API_KEY not found in environment variables")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for TorchXRayVision models"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_array = skimage.color.rgb2gray(img_array)
        
        # Normalize to 0-255 range
        img_array = (img_array * 255).astype(np.uint8)
        
        # Apply TorchXRayVision preprocessing
        img_array = xrv.datasets.normalize(img_array, 255)
        
        # Resize to 224x224 for DenseNet or 512x512 for ResNet
        img_224 = skimage.transform.resize(img_array, (224, 224))
        img_512 = skimage.transform.resize(img_array, (512, 512))
        
        # Convert to torch tensors
        img_224_tensor = torch.FloatTensor(img_224).unsqueeze(0).unsqueeze(0)
        img_512_tensor = torch.FloatTensor(img_512).unsqueeze(0).unsqueeze(0)
        
        return {
            'densenet_input': img_224_tensor,
            'resnet_input': img_512_tensor
        }
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def analyze_xray_with_models(models, processed_images):
    """Analyze X-ray using TorchXRayVision models"""
    try:
        results = {}
        
        # Analyze with DenseNet
        with torch.no_grad():
            densenet_output = models['densenet'](processed_images['densenet_input'])
            densenet_probs = torch.sigmoid(densenet_output).cpu().numpy()[0]
            
        # Analyze with ResNet
        with torch.no_grad():
            resnet_output = models['resnet'](processed_images['resnet_input'])
            resnet_probs = torch.sigmoid(resnet_output).cpu().numpy()[0]
        
        # Get pathology labels
        pathologies = models['densenet'].pathologies
        
        # Combine results (average of both models)
        combined_probs = (densenet_probs + resnet_probs) / 2
        
        # Create findings dictionary
        findings = {}
        for i, pathology in enumerate(pathologies):
            findings[pathology] = {
                'probability': float(combined_probs[i]),
                'densenet_prob': float(densenet_probs[i]),
                'resnet_prob': float(resnet_probs[i]),
                'likely': combined_probs[i] > 0.5
            }
        
        return findings
    except Exception as e:
        st.error(f"Error analyzing X-ray: {str(e)}")
        return None

def generate_clinical_report(findings, patient_info, groq_client):
    """Generate clinical report using Groq"""
    
    # Prepare findings summary for the prompt
    significant_findings = []
    normal_findings = []
    
    for pathology, data in findings.items():
        if data['probability'] > 0.3:  # Threshold for mentioning in report
            if data['probability'] > 0.5:
                significant_findings.append(f"- {pathology}: {data['probability']:.1%} probability (POSITIVE)")
            else:
                significant_findings.append(f"- {pathology}: {data['probability']:.1%} probability (BORDERLINE)")
        else:
            normal_findings.append(f"- {pathology}: {data['probability']:.1%} probability (NEGATIVE)")
    
    findings_text = "SIGNIFICANT FINDINGS:\n" + "\n".join(significant_findings)
    findings_text += "\n\nNEGATIVE/NORMAL FINDINGS:\n" + "\n".join(normal_findings[:5])  # Limit normal findings
    
    prompt = f"""
    You are an expert radiologist writing a comprehensive chest X-ray report. Based on the AI model analysis results below, generate a professional radiology report following standard medical format.

    PATIENT INFORMATION:
    - Name: {patient_info.get('name', 'N/A')}
    - Age/Gender: {patient_info.get('age', 'N/A')}/{patient_info.get('gender', 'N/A')}
    - Date: {patient_info.get('date', 'N/A')}

    AI MODEL ANALYSIS RESULTS:
    {findings_text}

    Please structure your report with the following sections:

    **TECHNIQUE:**
    Comment on the X-ray technique and quality

    **COMPARISON:**
    State if prior studies are available (assume none for this case)

    **FINDINGS:**

    *Lungs and Airways:*
    Describe lung field findings based on the AI analysis, focusing on:
    - Consolidation, infiltrates, or opacity findings
    - Pneumonia or infection indicators
    - Atelectasis or lung collapse
    - Pleural effusion presence

    *Heart and Mediastinum:*
    Describe cardiac and mediastinal findings:
    - Cardiomegaly assessment
    - Mediastinal contours
    - Any enlarged cardiac silhouette

    *Bones and Soft Tissues:*
    Comment on:
    - Rib fractures or bone lesions
    - Soft tissue findings
    - Any skeletal abnormalities

    *Other:*
    - Medical devices if any
    - Other significant findings

    **IMPRESSION:**
    Provide a clear, concise clinical impression based on the findings. If significant pathology is detected (>50% probability), state it clearly. If findings are borderline (30-50%), mention as "possible" or "cannot exclude". For normal studies, state "No acute cardiopulmonary abnormality."

    Use proper medical terminology and maintain professional radiological language throughout. Be conservative in your interpretations - if AI confidence is borderline, reflect that uncertainty appropriately.
    """

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            max_tokens=1500,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error generating report with Groq: {str(e)}")
        return None

def create_findings_visualization(findings):
    """Create a visual representation of findings"""
    st.subheader("🔬 AI Model Analysis Results")
    
    # Sort findings by probability
    sorted_findings = sorted(findings.items(), key=lambda x: x[1]['probability'], reverse=True)
    
    # Create columns for metrics
    cols = st.columns(4)
    high_prob_findings = [f for f, d in sorted_findings if d['probability'] > 0.5]
    
    with cols[0]:
        st.metric("High Probability Findings", len(high_prob_findings))
    with cols[1]:
        st.metric("Total Pathologies Analyzed", len(findings))
    with cols[2]:
        max_prob = max(d['probability'] for d in findings.values())
        st.metric("Highest Probability", f"{max_prob:.1%}")
    with cols[3]:
        avg_prob = np.mean([d['probability'] for d in findings.values()])
        st.metric("Average Probability", f"{avg_prob:.1%}")
    
    st.write("---")
    
    # Display top findings
    st.subheader("📊 Top Findings (by probability)")
    
    for pathology, data in sorted_findings[:10]:  # Show top 10
        prob = data['probability']
        
        # Determine styling based on probability
        if prob > 0.5:
            box_class = "finding-box abnormal-finding"
            status = "🔴 POSITIVE"
        elif prob > 0.3:
            box_class = "finding-box"
            status = "🟡 BORDERLINE"
        else:
            box_class = "finding-box"
            status = "🟢 NEGATIVE"
        
        st.markdown(f'''
        <div class="{box_class}">
            <strong>{pathology}</strong> - {status}<br>
            <strong>Combined Probability:</strong> {prob:.1%}<br>
            <small>DenseNet: {data['densenet_prob']:.1%} | ResNet: {data['resnet_prob']:.1%}</small>
        </div>
        ''', unsafe_allow_html=True)
        
        # Add progress bar
        st.progress(prob)

def generate_full_report(clinical_report, findings, patient_info):
    """Generate complete structured report"""
    
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# **CHEST X-RAY COMPREHENSIVE REPORT**

## **Patient Information**
- **Name:** {patient_info.get('name', 'N/A')}
- **Age/Gender:** {patient_info.get('age', 'N/A')} / {patient_info.get('gender', 'N/A')}
- **Patient ID:** {patient_info.get('patient_id', 'N/A')}
- **Date of Examination:** {patient_info.get('date', current_date)}
- **Referring Physician:** {patient_info.get('physician', 'N/A')}

## **Type of Imaging**
- **Modality:** Digital Chest X-ray
- **View:** {patient_info.get('view', 'Anteroposterior (AP)')}
- **Purpose:** Comprehensive chest evaluation

---

## **CLINICAL REPORT**

{clinical_report}

---

## **AI MODEL ANALYSIS SUMMARY**

**Models Used:**
- TorchXRayVision DenseNet121 (trained on CheXpert, MIMIC-CXR, NIH, etc.)
- TorchXRayVision ResNet50 (trained on multiple chest X-ray datasets)

**High Confidence Findings (>50% probability):**
"""
    
    high_conf_findings = [f"- {path}: {data['probability']:.1%}" 
                         for path, data in findings.items() 
                         if data['probability'] > 0.5]
    
    if high_conf_findings:
        report += "\n".join(high_conf_findings)
    else:
        report += "- No high-confidence positive findings detected"
    
    report += f"""

**Borderline Findings (30-50% probability):**
"""
    
    borderline_findings = [f"- {path}: {data['probability']:.1%}" 
                          for path, data in findings.items() 
                          if 0.3 <= data['probability'] <= 0.5]
    
    if borderline_findings:
        report += "\n".join(borderline_findings)
    else:
        report += "- No borderline findings detected"
    
    report += f"""

---

## **TECHNICAL DETAILS**
- **Report Generated:** {current_date}
- **Analysis Method:** Deep Learning Classification + AI Report Generation
- **Model Confidence:** Combined probability from ensemble of specialized chest X-ray CNNs
- **Processing:** TorchXRayVision preprocessing pipeline

---

**DISCLAIMER:** This report is generated using AI assistance for preliminary analysis. All findings must be validated by a qualified radiologist before making clinical decisions. This tool is intended for educational and screening purposes only.
"""
    
    return report

def main():
    st.markdown('<h1 class="main-header">🩺 AI-Powered Chest X-Ray Analysis</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong> This application uses specialized chest X-ray AI models (TorchXRayVision) 
        for preliminary analysis. All results must be reviewed by a qualified radiologist before making clinical decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🔬 Loading specialized chest X-ray models..."):
        models = load_xray_models()
        groq_client = init_groq_client()
    
    if not models or not groq_client:
        st.error("Failed to load required models. Please check your setup.")
        st.stop()
    
    st.markdown("""
    <div class="success-box">
        <strong>✅ Models Loaded Successfully!</strong><br>
        • TorchXRayVision DenseNet121 (trained on CheXpert, MIMIC-CXR, NIH)<br>
        • TorchXRayVision ResNet50 (multi-dataset training)<br>
        • Groq LLaMA 3.1 for report generation
    </div>
    """, unsafe_allow_html=True)
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Upload Chest X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear chest X-ray image for AI analysis"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray", use_column_width=True)
    
    with col2:
        st.subheader("👤 Patient Information")
        
        with st.form("patient_info_form"):
            patient_name = st.text_input("Patient Name")
            
            col_age, col_gender = st.columns(2)
            with col_age:
                patient_age = st.number_input("Age", min_value=0, max_value=150, value=None)
            with col_gender:
                patient_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
            
            patient_id = st.text_input("Patient ID")
            exam_date = st.date_input("Examination Date", datetime.now().date())
            physician = st.text_input("Referring Physician")
            view_type = st.selectbox("X-Ray View", ["AP (Anteroposterior)", "PA (Posteroanterior)"])
            
            analyze_button = st.form_submit_button("🔍 Analyze X-Ray", use_container_width=True)
    
    # Analysis section
    if uploaded_file and analyze_button:
        patient_info = {
            'name': patient_name or 'Anonymous Patient',
            'age': str(patient_age) if patient_age else 'N/A',
            'gender': patient_gender or 'N/A',
            'patient_id': patient_id or 'N/A',
            'date': exam_date.strftime("%Y-%m-%d") if exam_date else 'N/A',
            'physician': physician or 'N/A',
            'view': view_type
        }
        
        with st.spinner("🧠 Running AI analysis... This may take a moment."):
            # Preprocess image
            processed_images = preprocess_image(image)
            
            if processed_images:
                # Analyze with models
                findings = analyze_xray_with_models(models, processed_images)
                
                if findings:
                    # Create findings visualization
                    create_findings_visualization(findings)
                    
                    st.write("---")
                    
                    # Generate clinical report
                    with st.spinner("📝 Generating clinical report..."):
                        clinical_report = generate_clinical_report(findings, patient_info, groq_client)
                    
                    if clinical_report:
                        # Display clinical report
                        st.markdown('<div class="report-section">', unsafe_allow_html=True)
                        st.markdown("## 📋 Clinical Report")
                        st.markdown(clinical_report)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Generate full report for download
                        full_report = generate_full_report(clinical_report, findings, patient_info)
                        
                        # Download button
                        st.download_button(
                            label="📄 Download Complete Report",
                            data=full_report,
                            file_name=f"chest_xray_report_{patient_info['name'].replace(' ', '_')}_{exam_date.strftime('%Y%m%d')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    else:
                        st.error("❌ Failed to generate clinical report.")
                else:
                    st.error("❌ Failed to analyze the image.")
            else:
                st.error("❌ Failed to preprocess the image.")
    
    # Sidebar with model information
    with st.sidebar:
        st.markdown("## 🤖 AI Models Used")
        st.markdown("""
        **TorchXRayVision Models:**
        - **DenseNet121**: Trained on CheXpert, MIMIC-CXR, NIH14, PadChest
        - **ResNet50**: Multi-dataset ensemble training
        
        **Pathologies Detected:**
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
        - Nodule/Mass
        - Hernia
        - Lung Lesion
        - Fracture
        
        **Report Generation:**
        - Groq LLaMA 3.1 8B Instant
        """)
        
        st.markdown("---")
        st.markdown("**⚡ Powered by Domain-Specific Medical AI**")

if __name__ == "__main__":
    main()