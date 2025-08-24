# MediScan AI - Professional Cancer Detection System

A professional medical-grade cancer detection system powered by TensorFlow and AI technology.

## Features

- **Multi-class Cancer Detection**: Supports 26 different cancer types across 8 categories
- **AI-Powered Analysis**: Uses advanced deep learning models for accurate predictions
- **Professional Medical Interface**: Clean, trustworthy design suitable for medical environments
- **Confidence Scoring**: Detailed confidence analysis with visual charts
- **AI Descriptions**: Gemini AI provides detailed explanations of detected conditions
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

## Supported Cancer Types

### Acute Lymphoblastic Leukemia (ALL)
- Benign, Early, Pre, Pro stages

### Brain Cancer
- Glioma, Meningioma, Pituitary Tumor

### Breast Cancer
- Benign, Malignant

### Cervical Cancer
- Dyskeratotic, Koilocytotic, Metaplastic, Parabasal, Superficial-Intermediate

### Kidney Cancer
- Normal, Tumor

### Lung & Colon Cancer
- Colon Adenocarcinoma, Colon Benign Tissue
- Lung Adenocarcinoma, Lung Benign Tissue, Lung SCC

### Lymphoma
- Chronic Lymphocytic Leukemia, Follicular Lymphoma, Mantle Cell Lymphoma

### Oral Cancer
- Normal, SCC (Squamous Cell Carcinoma)

## Installation

1. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Update the model path in `app.py`:
```python
MODEL_PATH = "path/to/your/best_model.h5"
