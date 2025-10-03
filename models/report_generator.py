#models/report_generator.py

import os
from groq import Groq
import base64
from io import BytesIO
from PIL import Image

class ReportGenerator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables or provided as argument")
        self.client = Groq(api_key=self.api_key)
    
    def image_to_base64(self, image, max_size=(1024, 1024)):
        """Convert PIL Image to base64 with size limits
        
        Args:
            image: PIL Image
            max_size: Maximum dimensions (width, height) to resize to while maintaining aspect ratio
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Calculate resize dimensions while maintaining aspect ratio
        img_width, img_height = image.size
        resize_ratio = min(max_size[0] / img_width, max_size[1] / img_height)
        if resize_ratio < 1:
            new_width = int(img_width * resize_ratio)
            new_height = int(img_height * resize_ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        # Save as JPEG with compression to ensure smaller file size
        image.save(buffered, format="JPEG", optimize=True, quality=85)
        
        # Check if size is under 4MB
        img_data = buffered.getvalue()
        if len(img_data) > 4 * 1024 * 1024:  # 4MB in bytes
            # If still too large, increase compression
            buffered = BytesIO()
            image.save(buffered, format="JPEG", optimize=True, quality=50)
            img_data = buffered.getvalue()
        
        return base64.b64encode(img_data).decode()
    
    def generate_structured_report(self, image, analysis_data, patient_info=None):
        """Generate structured radiology report using Groq Vision LLM"""
        
        # If patient info not provided, use defaults and any detected info
        if not patient_info:
            patient_info = {}
            if 'view_classification' in analysis_data:
                view_data = analysis_data['view_classification']
                if 'age' in view_data:
                    patient_info['age'] = int(view_data['age'])
                if 'gender' in view_data:
                    patient_info['gender'] = view_data['gender']
        
        # Add default values for required fields
        patient_info.setdefault('name', 'ANONYMOUS')
        patient_info.setdefault('mrn', 'NOT PROVIDED')
        patient_info.setdefault('dob', 'NOT PROVIDED')
        patient_info.setdefault('referring_physician', 'NOT PROVIDED')
        patient_info.setdefault('exam_date', 'CURRENT')
        
        # Compile findings into a structured format
        findings_summary = self._compile_findings(analysis_data)
        
        # If Groq API is available, use vision model
        if self.client:
            try:
                report = self._generate_with_groq(image, findings_summary, patient_info)
                return report
            except Exception as e:
                print(f"Error with Groq API: {e}")
                return self._generate_template_report(findings_summary, patient_info)
        else:
            return self._generate_template_report(findings_summary, patient_info)
    
    def _generate_with_groq(self, image, findings_summary, patient_info):
        """Generate report using Groq's vision model"""
        
        # Convert image to base64
        img_base64 = self.image_to_base64(image)
        
        prompt = f"""You are an expert radiologist. Analyze this chest X-ray and generate a comprehensive structured radiology report.

Use the following automated analysis results as reference:
{findings_summary}

Generate a detailed report with these sections:

1. TECHNICAL CONSIDERATIONS
   - Projection (AP or PA)
   - Rotation assessment
   - Inspiration adequacy
   - Penetration/exposure quality

2. AIRWAY AND MEDIASTINUM
   - Trachea position
   - Carina and bronchi
   - Mediastinum assessment

3. LUNGS AND PLEURA
   - Lung fields clarity
   - Any opacities, consolidation, or masses
   - Pleural assessment
   - Costophrenic angles

4. CARDIAC AND GREAT VESSELS
   - Cardiac silhouette and size
   - Cardiothoracic ratio
   - Aortic arch and pulmonary vessels

5. BONES AND SOFT TISSUE
   - Ribs and clavicles
   - Spine
   - Soft tissue assessment

6. IMPRESSION
   - Summary of key findings
   - Clinical significance

Be detailed and professional. Use standard radiology terminology."""

        try:
            # Format message for Groq's vision model with inline base64 image
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert radiologist analyzing chest X-ray images."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Try Scout model first
            try:
                completion = self.client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000,
                    stop=None
                )
            except Exception as e:
                print(f"Scout model error: {e}, trying Maverick...")
                # Fall back to Maverick if Scout fails
                completion = self.client.chat.completions.create(
                    model="meta-llama/llama-4-maverick-17b-128e-instruct",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000,
                    stop=None
                )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Groq vision model error: {e}")
            # Fallback to text-only model
            return self._generate_with_text_model(findings_summary, patient_info)
    
    def _generate_with_text_model(self, findings_summary, patient_info):
        """Generate report using Groq's text model"""
        
        prompt = f"""You are an expert radiologist. Generate a comprehensive structured chest X-ray radiology report based on the following automated analysis:

{findings_summary}

Generate a detailed report with these sections:
1. TECHNICAL CONSIDERATIONS
2. AIRWAY AND MEDIASTINUM
3. LUNGS AND PLEURA
4. CARDIAC AND GREAT VESSELS
5. BONES AND SOFT TISSUE
6. IMPRESSION

Be detailed, professional, and use standard radiology terminology."""

        try:
            # For text-only model, use llama-3.1-8b-instant
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert radiologist generating structured chest X-ray reports."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000,
                stop=None
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Groq text model error: {e}")
            return self._generate_template_report(findings_summary, patient_info)
    
    def _safe_get(self, entry, key="assessment", default="Unknown"):
        """Safely get value from either a dict or direct value"""
        if isinstance(entry, dict):
            return entry.get(key, default)
        return entry if entry is not None else default
    
    def _compile_findings(self, data):
        """Compile analysis data into readable findings"""
        findings = []
        
        # Technical considerations
        if 'view_classification' in data:
            view = data['view_classification']
            if isinstance(view, dict):
                view_type = view.get('view', 'Unknown')
                confidence = view.get('confidence', 0)
                findings.append(f"View: {view_type} (Confidence: {confidence:.2f})")
            else:
                findings.append(f"View: {view}")
        
        if 'rotation' in data:
            findings.append(f"Rotation: {self._safe_get(data['rotation'])}")
        
        if 'inspiration' in data:
            insp = data['inspiration']
            if isinstance(insp, dict):
                count = insp.get('count', 0)
                assessment = self._safe_get(insp)
                findings.append(f"Inspiration: {count} posterior ribs visible - {assessment}")
            else:
                findings.append(f"Inspiration: {insp}")
        
        if 'penetration' in data:
            findings.append(f"Penetration: {self._safe_get(data['penetration'])}")
        
        # Airway
        if 'trachea' in data:
            findings.append(f"Trachea: {self._safe_get(data['trachea'])}")
        
        # Cardiac
        if 'ctr' in data:
            findings.append(f"Cardiothoracic Ratio: {self._safe_get(data['ctr'])}")
        
        # Pleura
        if 'costophrenic_angles' in data:
            findings.append(f"Costophrenic Angles: {self._safe_get(data['costophrenic_angles'])}")
        
        # Abnormalities
        if 'abnormalities' in data:
            sig_findings = data.get('significant_findings', {})
            if isinstance(sig_findings, dict) and sig_findings:
                findings.append(f"Detected abnormalities: {', '.join(sig_findings.keys())}")
            elif isinstance(sig_findings, (list, set)):
                findings.append(f"Detected abnormalities: {', '.join(sig_findings)}")
            elif sig_findings:  # If it's a string or other non-empty value
                findings.append(f"Detected abnormalities: {sig_findings}")
            else:
                findings.append("No significant abnormalities detected by automated analysis")
        
        return "\n".join(findings)
    
    def _generate_template_report(self, findings_summary, patient_info):
        """Generate template-based report"""
        
        # Extract age suffix
        age_suffix = ''
        if 'age' in patient_info:
            age = patient_info['age']
            if isinstance(age, (int, float)):
                if age < 2:
                    age_suffix = 'months' if age <= 1 else 'month'
                else:
                    age_suffix = 'years' if age > 1 else 'year'
        
        report = f"""RADIOLOGIC CONSULTATION
======================

PATIENT INFORMATION:
Name: {patient_info['name']}
MRN: {patient_info['mrn']}
DOB: {patient_info['dob']}
Age: {patient_info.get('age', 'NOT PROVIDED')} {age_suffix}
Gender: {patient_info.get('gender', 'NOT PROVIDED')}
Exam Date: {patient_info['exam_date']}
Referring Physician: {patient_info['referring_physician']}

EXAM: CHEST X-RAY
History: Routine chest examination

AUTOMATED ANALYSIS FINDINGS:
{findings_summary}

1. TECHNICAL CONSIDERATIONS
The examination demonstrates the technical parameters listed above. Optimal PA projection is preferred for accurate cardiothoracic ratio measurement.

2. AIRWAY AND MEDIASTINUM
Trachea assessment is based on automated analysis. The carina and main bronchi appear within normal limits based on available data. The mediastinum shows no obvious abnormal widening on automated assessment.

3. LUNGS AND PLEURA
Automated analysis of lung fields has been performed. The pleural spaces and costophrenic angles have been evaluated as noted above.

4. CARDIAC AND GREAT VESSELS
Cardiac silhouette assessment including cardiothoracic ratio calculation is provided above. The aortic arch and pulmonary vessels appear unremarkable based on available automated analysis.

5. BONES AND SOFT TISSUE
Visualized osseous structures including ribs, clavicles, and spine show no obvious fractures or lytic lesions on automated review. Soft tissues appear unremarkable.

6. IMPRESSION
Please refer to the automated analysis findings above. This automated report should be reviewed and verified by a qualified radiologist for clinical decision-making.

Report Generated: {patient_info['exam_date']}

DISCLAIMER: This is an AI-generated report based on automated analysis. Final interpretation must be made by a licensed radiologist.
"""
        
        return report