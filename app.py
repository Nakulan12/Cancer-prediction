from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
import base64
import google.generativeai as genai

# Flask app
app = Flask(__name__)

# ✅ Gemini API (Replace with your key)
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ✅ Labels (same order as training)
labels = [
    "all_all_benign", "all_all_early", "all_all_pre", "all_all_pro",
    "brain_cancer_brain_glioma", "brain_cancer_brain_menin", "brain_cancer_brain_tumor",
    "breast_cancer_breast_benign", "breast_cancer_breast_malignant",
    "cervical_cancer_cervix_dyk", "cervical_cancer_cervix_koc", "cervical_cancer_cervix_mep",
    "cervical_cancer_cervix_pab", "cervical_cancer_cervix_sfi",
    "kidney_cancer_kidney_normal", "kidney_cancer_kidney_tumor",
    "lung_and_colon_cancer_colon_aca", "lung_and_colon_cancer_colon_bnt",
    "lung_and_colon_cancer_lung_aca", "lung_and_colon_cancer_lung_bnt",
    "lung_and_colon_cancer_lung_scc",
    "lymphoma_lymph_cll", "lymphoma_lymph_fl", "lymphoma_lymph_mcl",
    "oral_cancer_oral_normal", "oral_cancer_oral_scc"
]

# ✅ Descriptive map
descriptive_map = {
    "all_all_benign": "Acute Lymphoblastic Leukemia (ALL) - Benign",
    "all_all_early": "Acute Lymphoblastic Leukemia (ALL) - Early",
    "all_all_pre": "Acute Lymphoblastic Leukemia (ALL) - Pre",
    "all_all_pro": "Acute Lymphoblastic Leukemia (ALL) - Pro",
    "brain_cancer_brain_glioma": "Brain Cancer - Glioma",
    "brain_cancer_brain_menin": "Brain Cancer - Meningioma",
    "brain_cancer_brain_tumor": "Brain Cancer - Pituitary Tumor",
    "breast_cancer_breast_benign": "Breast Cancer - Benign",
    "breast_cancer_breast_malignant": "Breast Cancer - Malignant",
    "cervical_cancer_cervix_dyk": "Cervical Cancer - Dyskeratotic",
    "cervical_cancer_cervix_koc": "Cervical Cancer - Koilocytotic",
    "cervical_cancer_cervix_mep": "Cervical Cancer - Metaplastic",
    "cervical_cancer_cervix_pab": "Cervical Cancer - Parabasal",
    "cervical_cancer_cervix_sfi": "Cervical Cancer - Superficial-Intermediate",
    "kidney_cancer_kidney_normal": "Kidney Cancer - Normal",
    "kidney_cancer_kidney_tumor": "Kidney Cancer - Tumor",
    "lung_and_colon_cancer_colon_aca": "Colon Cancer - Adenocarcinoma",
    "lung_and_colon_cancer_colon_bnt": "Colon Cancer - Benign Tissue",
    "lung_and_colon_cancer_lung_aca": "Lung Cancer - Adenocarcinoma",
    "lung_and_colon_cancer_lung_bnt": "Lung Cancer - Benign Tissue",
    "lung_and_colon_cancer_lung_scc": "Lung Cancer - SCC (Squamous Cell Carcinoma)",
    "lymphoma_lymph_cll": "Lymphoma - Chronic Lymphocytic Leukemia",
    "lymphoma_lymph_fl": "Lymphoma - Follicular Lymphoma",
    "lymphoma_lymph_mcl": "Lymphoma - Mantle Cell Lymphoma",
    "oral_cancer_oral_normal": "Oral Cancer - Normal",
    "oral_cancer_oral_scc": "Oral Cancer - SCC (Squamous Cell Carcinoma)"
}

# ✅ Load model safely from 'model/best_model.h5'
MODEL_PATH = os.path.join("model", "best_model.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please place your model in the 'model' folder.")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ✅ Risk assessment function
def get_risk_assessment(confidence_score):
    if confidence_score >= 80:
        return {"level": "High Confidence", "message": "Immediate medical consultation recommended"}
    elif confidence_score >= 60:
        return {"level": "Moderate Confidence", "message": "Further testing may be required"}
    else:
        return {"level": "Low Confidence", "message": "Results inconclusive, additional analysis needed"}

# ✅ Main route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence_score = None
    description = None
    chart_url = None
    uploaded_image_url = None
    risk_assessment = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            os.makedirs("static", exist_ok=True)
            filepath = os.path.join("static", file.filename)
            file.save(filepath)
            uploaded_image_url = filepath

            # Preprocess image
            img = Image.open(filepath).convert("RGB").resize((224, 224))
            img_array = image.img_to_array(img) / 255.0
            input_array = np.expand_dims(img_array, axis=0)

            # Prediction
            pred = model.predict(input_array)[0]
            pred_index = np.argmax(pred)
            predicted_label = labels[pred_index]
            confidence_score = pred[pred_index] * 100
            prediction = descriptive_map.get(predicted_label, "Unknown class")
            risk_assessment = get_risk_assessment(confidence_score)

            # Gemini AI description
            try:
                prompt = f"Give a short 5-line description about {prediction} in simple terms."
                gemini_response = gemini_model.generate_content(prompt)
                description = gemini_response.text.strip()
            except:
                description = "AI description unavailable. Please consult a medical professional."

            # Confidence chart
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh([descriptive_map.get(label, label) for label in labels], pred, color='lightseagreen')
            ax.set_xlim(0, 1)
            ax.set_xlabel("Confidence")
            ax.set_title("Model Confidence", fontsize=14)
            ax.invert_yaxis()

            for i, val in enumerate(pred):
                ax.text(val + 0.01, i, f"{val*100:.2f}%", va='center')

            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            buf.seek(0)
            chart_url = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence_score,
        description=description,
        chart_url=chart_url,
        uploaded_image_url=uploaded_image_url,
        risk_assessment=risk_assessment
    )

if __name__ == "__main__":
    app.run(debug=True)
