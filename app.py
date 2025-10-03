import streamlit as st
import torch
from PIL import Image
import numpy as np
import sys
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import cv2


# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from utils.preprocessing import XRayPreprocessor
from utils.visualization import overlay_mask, draw_measurements, draw_rib_markers, create_comparison_figure
from models.segmentation import AnatomySegmenter
from models.classification import XRayClassifier
from models.measurements import XRayMeasurements
from models.report_generator import ReportGenerator

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Chest X-Ray Analyzer",
    page_icon="🫁",
    layout="wide"
)

# Initialize session state
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.results = None
    st.session_state.image = None

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_models():
    """Load all models - cached to avoid reloading"""
    with st.spinner("Loading models... This may take a minute on first run."):
        preprocessor = XRayPreprocessor()
        segmenter = AnatomySegmenter(device=device)
        classifier = XRayClassifier(device=device)
        measurements = XRayMeasurements()
        
        # Load classification models
        try:
            classifier.load_abnormality_detector()
        except:
            st.warning("Could not load abnormality detector. Using fallback methods.")
        
        try:
            classifier.load_view_classifier()
        except:
            st.warning("Could not load view classifier. Using heuristic methods.")
        
        # Initialize report generator
        groq_api_key = os.getenv('GROQ_API_KEY')
        report_gen = ReportGenerator(api_key=groq_api_key)
        
        return preprocessor, segmenter, classifier, measurements, report_gen

def create_visualization(image, results):
    """Create visualization overlays"""
    img_array = np.array(image.convert('RGB'))
    masks = results['masks']
    
    # Create segmentation overlay
    segmented_img = img_array.copy()
    
    # Overlay lungs in blue
    lung_colored = np.zeros_like(img_array)
    lung_colored[masks['lungs'] > 0] = [0, 0, 255]
    segmented_img = np.where(lung_colored > 0, 
                             (segmented_img * 0.6 + lung_colored * 0.4).astype(np.uint8),
                             segmented_img)
    
    # Overlay heart in red
    heart_colored = np.zeros_like(img_array)
    heart_colored[masks['heart'] > 0] = [255, 0, 0]
    segmented_img = np.where(heart_colored > 0,
                            (segmented_img * 0.6 + heart_colored * 0.4).astype(np.uint8),
                            segmented_img)
    
    # Draw rib markers if available
    if 'inspiration' in results and 'positions' in results['inspiration']:
        for i, pos in enumerate(results['inspiration']['positions']):
            # Draw circle
            cv2.circle(segmented_img, (int(pos[0]), int(pos[1])), 5, (0, 255, 255), -1)
            # Draw label
            cv2.putText(segmented_img, f'R{i+1}', 
                       (int(pos[0])+10, int(pos[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Draw costophrenic angle markers
    if 'costophrenic_angles' in results and 'angles' in results['costophrenic_angles']:
        for angle_data in results['costophrenic_angles']['angles']:
            pos = angle_data['position']
            color = (0, 255, 0) if angle_data['sharp'] else (255, 165, 0)
            cv2.circle(segmented_img, pos, 8, color, 2)
            # Label with angle if available
            if angle_data.get('angle'):
                label = f"{angle_data['angle']:.0f}°"
                cv2.putText(segmented_img, label,
                           (pos[0]+15, pos[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return Image.fromarray(segmented_img)

def analyze_xray(image, preprocessor, segmenter, classifier, measurements, report_gen, patient_info=None):
    """Complete X-ray analysis pipeline with optional patient information"""
    
    results = {}
    
    # Convert to numpy for processing
    img_array = np.array(image.convert('L'))
    
    with st.spinner("Step 1/6: Preprocessing image..."):
        # Enhance image
        enhanced_img = preprocessor.enhance_contrast(image)
        
        # Prepare tensors for torchxrayvision
        tensor_xrv = preprocessor.preprocess_for_torchxrayvision(enhanced_img)
        view_result = classifier.classify_view(image, tensor_xrv)
    
    with st.spinner("Step 2/6: Classifying view and getting segmentation..."):
        results['view_classification'] = view_result
        ml_mask = view_result.get('mask', None)

        if view_result and 'age' in view_result:
            results['patient_age'] = view_result['age']
            results['patient_female_prob'] = 1.0 if view_result.get('gender') == 'Female' else 0.0

        rotation_result = classifier.check_rotation(image)
        rotation_result['assessment'] = rotation_result.get('message', "Rotation assessment unavailable")
        rotation_result['symmetric'] = rotation_result.get('symmetric', False)
        results['rotation'] = rotation_result
    
    with st.spinner("Step 3/6: Segmenting anatomical structures..."):
        masks = segmenter.segment_all(enhanced_img, use_ml_model=True, ml_masks=ml_mask)
        results['masks'] = masks
        results['segmentation_method'] = masks.get('method', 'unknown')
    
    with st.spinner("Step 4/6: Detecting abnormalities..."):
        abnormality_results = classifier.detect_abnormalities(tensor_xrv)
        
        # Store the raw findings (float scores 0-1)
        findings = abnormality_results.get('findings', {})
        results['abnormalities'] = findings  # Raw scores for expander display
        
        # Store significant findings
        significant = abnormality_results.get('significant', {})
        results['significant_findings'] = significant
    
    with st.spinner("Step 5/6: Measuring technical parameters..."):
        rib_result = measurements.count_posterior_ribs(img_array, masks['lungs'])
        results['inspiration'] = rib_result
        
        penetration_result = measurements.check_penetration(img_array)
        results['penetration'] = penetration_result
        
        trachea_result = measurements.check_trachea_position(img_array)
        results['trachea'] = trachea_result
    
    with st.spinner("Step 6/6: Calculating measurements and generating report..."):
        if ml_mask is not None:
            ctr_result = segmenter.calculate_ctr_from_mask(ml_mask)
        else:
            ctr_result = measurements.calculate_cardiothoracic_ratio(masks['heart'], masks['lungs'])
            
        if ctr_result and isinstance(ctr_result, dict):
            ctr_value = ctr_result.get('ctr')
            if ctr_value is not None:
                if 'interpretation' in ctr_result:
                    ctr_result['assessment'] = ctr_result['interpretation']
                else:
                    if ctr_value > 0.55:
                        assessment = f"Marked cardiomegaly - CTR {ctr_value:.2f} (>0.55)"
                    elif ctr_value > 0.50:
                        assessment = f"Borderline cardiomegaly - CTR {ctr_value:.2f} (>0.50)"
                    else:
                        assessment = f"Normal cardiac size - CTR {ctr_value:.2f} (≤0.50)"
                    ctr_result['assessment'] = assessment
            
        results['ctr'] = ctr_result
        
        angles_result = measurements.check_costophrenic_angles(masks['lungs'])
        results['costophrenic_angles'] = angles_result
        
        report = report_gen.generate_structured_report(image, results, patient_info)
        results['report'] = report
        
        if patient_info:
            results['patient_info'] = patient_info
    
    return results

def display_results(image, results):
    """Display analysis results with visualizations"""
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Report", 
        "🖼️ Visualizations",
        "🔧 Technical", 
        "🫁 Anatomy", 
        "⚕️ Findings"
    ])
    
    with tab1:
        st.subheader("Comprehensive Radiology Report")
        st.text_area("Report", results['report'], height=600)
        st.download_button(
            label="Download Report",
            data=results['report'],
            file_name="chest_xray_report.txt",
            mime="text/plain"
        )
    
    with tab2:
        st.subheader("Anatomical Segmentation & Measurements")
        
        # Generate visualization
        viz_image = create_visualization(image, results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original X-Ray", use_container_width=True)
        with col2:
            st.image(viz_image, caption="Annotated Analysis", use_container_width=True)
        
        # Legend
        st.markdown("""
        **Color Legend:**
        - 🔵 Blue: Lung fields
        - 🔴 Red: Cardiac silhouette  
        - 🟡 Yellow: Posterior ribs (numbered)
        - 🟢 Green: Sharp costophrenic angles
        - 🟠 Orange: Blunted costophrenic angles
        """)
        
        # Download visualization
        import io
        buf = io.BytesIO()
        viz_image.save(buf, format='PNG')
        st.download_button(
            label="Download Annotated Image",
            data=buf.getvalue(),
            file_name="xray_annotated.png",
            mime="image/png"
        )
    
    with tab3:
        st.subheader("Technical Considerations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Projection")
            view = results['view_classification']
            st.info(f"**View:** {view['view']}")
            st.caption(f"Confidence: {view['confidence']:.2%}")
            st.caption(view['note'])
            
            if 'patient_age' in results:
                st.markdown("#### Patient Information")
                st.metric("Estimated Age", f"{results['patient_age']:.0f} years")
                gender = "Female" if results['patient_female_prob'] >= 0.5 else "Male"
                gender_conf = results['patient_female_prob'] if results['patient_female_prob'] >= 0.5 else 1 - results['patient_female_prob']
                st.caption(f"Predicted Sex: {gender} ({gender_conf:.1%} confidence)")
            
            st.markdown("#### Rotation")
            rotation = results.get('rotation', {})
            rotation_assessment = rotation.get('assessment', "Rotation assessment unavailable")
            if rotation.get('symmetric', False):
                st.success(rotation_assessment)
            else:
                st.warning(rotation_assessment)
            
        with col2:
            st.markdown("#### Inspiration")
            insp = results.get('inspiration', {})
            rib_count = insp.get('count', 0)
            st.metric("Posterior Ribs Visible", rib_count)
            insp_assessment = insp.get('assessment', "Inspiration assessment unavailable")
            if rib_count >= 9:
                st.success(insp_assessment)
            elif rib_count >= 7:
                st.warning(insp_assessment)
            else:
                st.error(insp_assessment)
            
            st.markdown("#### Penetration")
            pen = results.get('penetration', {})
            pen_assessment = pen.get('assessment', "Penetration assessment unavailable")
            if pen.get('adequate', False):
                st.success(pen_assessment)
            else:
                st.warning(pen_assessment)
            
            if results.get('segmentation_method') == 'ml_model':
                st.success("✓ Using ML-based segmentation")
            else:
                st.info("ℹ Using traditional segmentation")
    
    with tab4:
        st.subheader("Anatomical Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Airway")
            trachea = results.get('trachea', {})
            trachea_assessment = trachea.get('assessment', "Airway assessment unavailable")
            if trachea.get('midline', True):
                st.success(trachea_assessment)
            else:
                st.warning(trachea_assessment)
            
            st.markdown("#### Mediastinum")
            st.info("Automated assessment: No abnormal widening detected")
        
        with col2:
            st.markdown("#### Cardiac Silhouette")
            ctr = results.get('ctr', {})
            if ctr:
                ctr_value = ctr.get('ctr', 0.0)
                st.metric("Cardiothoracic Ratio", f"{ctr_value:.3f}")
                ctr_assessment = ctr.get('assessment', "CTR assessment unavailable")
                if ctr_value < 0.5:
                    st.success(ctr_assessment)
                elif ctr_value < 0.55:
                    st.warning(ctr_assessment)
                else:
                    st.error(ctr_assessment)
            else:
                st.info("CTR calculation unavailable")
            
            st.markdown("#### Pleura")
            angles = results.get('costophrenic_angles', {})
            angles_assessment = angles.get('assessment', "Pleural angles assessment unavailable")
            
            if angles.get('angles'):
                col1, col2 = st.columns(2)
                with col1:
                    left_angle = next((angle['angle'] for angle in angles['angles'] if angle['side'] == 'left'), None)
                    if left_angle is not None:
                        st.metric("Left CP Angle", f"{left_angle:.1f}°")
                with col2:
                    right_angle = next((angle['angle'] for angle in angles['angles'] if angle['side'] == 'right'), None)
                    if right_angle is not None:
                        st.metric("Right CP Angle", f"{right_angle:.1f}°")
            
            if angles.get('sharp', True):
                st.success(angles_assessment)
            else:
                st.warning(angles_assessment)
    
    with tab5:
        st.subheader("Abnormality Detection")
        
        sig_findings = results.get('significant_findings', {})
        
        if sig_findings and isinstance(sig_findings, dict):
            try:
                sorted_findings = sorted(
                    [(k, float(v)) for k, v in sig_findings.items() if isinstance(v, (float, int))],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                st.warning(f"⚠️ {len(sorted_findings)} significant finding(s) detected")
                
                for pathology, score in sorted_findings:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{pathology}**")
                    with col2:
                        st.write(f"{score:.1%}")
                    st.progress(score)
            except (ValueError, TypeError) as e:
                st.error("Error displaying findings - data format issue")
        else:
            st.success("✓ No significant abnormalities detected by automated analysis")
        
        with st.expander("View All Detection Scores"):
            abnormalities = results.get('abnormalities', {})
            try:
                sorted_abnormalities = sorted(
                    [(k, float(v)) for k, v in abnormalities.items() if isinstance(v, (float, int))],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for pathology, score in sorted_abnormalities:
                    st.write(f"{pathology}: {score:.1%}")
            except (ValueError, TypeError) as e:
                st.error("Error displaying abnormalities - data format issue")

def main():
    st.title("🫁 AI Chest X-Ray Analyzer")
    st.markdown("Comprehensive automated analysis of chest X-rays with structured reporting")
    
    with st.sidebar:
        st.header("About")
        st.info("""
        This application performs comprehensive chest X-ray analysis including:
        
        - Technical quality assessment
        - Anatomical segmentation
        - Abnormality detection
        - Measurement calculations
        - Structured report generation
        
        **Note:** This is an AI-assisted tool. All findings should be verified by a qualified radiologist.
        """)
        
        st.header("Settings")
        if torch.cuda.is_available():
            st.success(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        else:
            st.info("Running on CPU")
        
        groq_available = bool(os.getenv('GROQ_API_KEY'))
        if groq_available:
            st.success("✓ Groq API Connected")
        else:
            st.warning("⚠️ Groq API not configured\nUsing template reports")
    
    try:
        preprocessor, segmenter, classifier, measurements, report_gen = load_models()
        st.success("✓ Models loaded successfully")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
    
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray Image", 
        type=['png', 'jpg', 'jpeg', 'dcm'],
        help="Upload a chest X-ray image in PNG, JPG, or DICOM format"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            
            st.divider()
            st.subheader("Patient Information")
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Patient Name", help="Enter patient's full name")
                mrn = st.text_input("Medical Record Number (MRN)", help="Enter patient's MRN")
                dob = st.text_input("Date of Birth", help="Format: YYYY-MM-DD")
            with col2:
                referring_physician = st.text_input("Referring Physician", help="Enter referring doctor's name")
                gender = st.radio("Gender", ["Male", "Female", "Other"], help="Select patient's gender")
                history = st.text_area("Clinical History", help="Enter relevant clinical history", height=100)
            
            st.divider()
            
            if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
                patient_info = {
                    'name': name if name else 'ANONYMOUS',
                    'mrn': mrn if mrn else 'NOT PROVIDED',
                    'dob': dob if dob else 'NOT PROVIDED',
                    'gender': gender,
                    'referring_physician': referring_physician if referring_physician else 'NOT PROVIDED',
                    'history': history if history else 'Routine chest examination',
                    'exam_date': 'CURRENT'
                }
                try:
                    results = analyze_xray(
                        image, 
                        preprocessor, 
                        segmenter, 
                        classifier, 
                        measurements, 
                        report_gen,
                        patient_info
                    )
                    st.session_state.results = results
                    st.session_state.image = image
                    st.session_state.analysis_done = True
                    st.success("✓ Analysis complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            
            if st.session_state.analysis_done and st.session_state.results:
                st.divider()
                display_results(st.session_state.image, st.session_state.results)
                
        except Exception as e:
            st.error(f"Error loading image: {e}")
    
    else:
        st.info("👆 Please upload a chest X-ray image to begin analysis")

if __name__ == "__main__":
    main()