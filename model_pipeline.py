import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from transformers import ViTForImageClassification, AutoModelForImageClassification
import gradio as gr
from PIL import Image
import numpy as np
import os

# --- Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Model Paths ---
BASE_MODEL_PATH = "/home/dheeraj/Documents/Swasth Avishkar Hackathon/Hackathon/Best_Base_Model.pth"
XRAY_MODEL_PATH = "/home/dheeraj/Documents/Swasth Avishkar Hackathon/Hackathon/best_xray_model_finetuned.pth"
MRI_MODEL_PATH = "/home/dheeraj/Documents/Swasth Avishkar Hackathon/Hackathon/best_mri_model_feature_extraction.pth"
CT_MODEL_PATH = "/home/dheeraj/Documents/Swasth Avishkar Hackathon/Hackathon/best_ct_model_swin.pth"

BASE_CLASS_NAMES = ['ct_scan', 'mri_scan', 'xray']
XRAY_CLASSES = sorted([
    'Breast_BIRAD-1', 'Breast_BIRAD-3', 'Breast_BIRAD-4', 'Breast_BIRAD-5', 'Chest_Atelectasis',
    'Chest_Cardiomegaly', 'Chest_Covid', 'Chest_Emphysema', 'Chest_No Finding', 'Chest_Pneumonia',
    'Chest_Tuberculosis', 'Limb_fractured', 'Limb_not fractured', 'Spine_Normal', 'Spine_Scoliosis',
    'Spine_Spondylolisthesis'
])
MRI_CLASSES = sorted([
    'Alzheimer_MildDemented', 'Alzheimer_ModerateDemented', 'Alzheimer_NonDemented', 'Alzheimer_VeryMildDemented',
    'BrainCancer_brain_glioma', 'BrainCancer_brain_menin', 'BrainCancer_brain_tumor', 'BrainTumor_glioma',
    'BrainTumor_meningioma', 'BrainTumor_notumor', 'BrainTumor_pituitary', 'BreastMRI_Benign', 'BreastMRI_Malignant'
])
CT_CLASSES = sorted([
    'Kidney_Cyst', 'Kidney_Normal', 'Kidney_Stone', 'Kidney_Tumor', 'Lung_Benign', 'Lung_Malignant',
    'Lung_Normal', 'SARS_COVID', 'SARS_non-COVID'
])

# --- Preprocessing Transforms ---
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
XRAY_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Model Loading Functions ---
def load_base_model():
    model = resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, len(BASE_CLASS_NAMES)))
    checkpoint = torch.load(BASE_MODEL_PATH, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

def load_xray_model():
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224', num_labels=len(XRAY_CLASSES), ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(XRAY_MODEL_PATH, map_location=device))
    return model.to(device).eval()

def load_mri_model():
    model = AutoModelForImageClassification.from_pretrained(
        'facebook/deit-base-patch16-224', num_labels=len(MRI_CLASSES), ignore_mismatched_sizes=True
    )
    model.classifier = nn.Sequential(nn.Dropout(0.5), model.classifier)
    model.load_state_dict(torch.load(MRI_MODEL_PATH, map_location=device))
    return model.to(device).eval()

def load_ct_model():
    model = AutoModelForImageClassification.from_pretrained(
        'microsoft/swin-base-patch4-window7-224', num_labels=len(CT_CLASSES), ignore_mismatched_sizes=True
    )
    model.classifier = nn.Sequential(nn.Dropout(0.5), model.classifier)
    model.load_state_dict(torch.load(CT_MODEL_PATH, map_location=device))
    return model.to(device).eval()

# --- Load all models at startup ---
print("Loading all models...")
model_base = load_base_model()
model_xray = load_xray_model()
model_mri = load_mri_model()
model_ct = load_ct_model()
print("All models loaded successfully!")


# --- CORE PREDICTION PIPELINE ### ---
def predict(image: Image.Image):
    """
    Handles prediction and provides valid outputs for all Gradio components in all cases.
    """
    if image is None:
        return "Please upload an image to begin.", {}

    try:
        image_rgb = image.convert("RGB")

        # Step 1: Base modality classification
        img_base = TRANSFORM(image_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_base = model_base(img_base)
            base_probs = torch.softmax(logits_base, dim=1).cpu().numpy().flatten()
            branch_idx = base_probs.argmax()
            branch_name = BASE_CLASS_NAMES[branch_idx]
            modality_conf = float(base_probs[branch_idx])

        # Step 2: Route to the correct specialized model
        if branch_name == 'xray':
            transform, model_spec, spec_classes = XRAY_TRANSFORM, model_xray, XRAY_CLASSES
        elif branch_name == 'mri_scan':
            transform, model_spec, spec_classes = TRANSFORM, model_mri, MRI_CLASSES
        elif branch_name == 'ct_scan':
            transform, model_spec, spec_classes = TRANSFORM, model_ct, CT_CLASSES
        else: # Should not happen, but good to have a fallback
            return f"Unknown modality: {branch_name}", {"Error": 0.0}

        # Step 3: Specialized disease prediction
        img_spec = transform(image_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model_spec(pixel_values=img_spec)
            logits_spec = output.logits
            probs_spec = torch.softmax(logits_spec, dim=1).cpu().numpy().flatten()

        # Format output dictionary with { "ClassName": float_probability }
        top3_idx = np.argsort(probs_spec)[-3:][::-1]
        disease_dict = {spec_classes[i].replace('_', ' ').title(): float(probs_spec[i]) for i in top3_idx}
        modality_str = f"{branch_name.replace('_', ' ').upper()} (Confidence: {modality_conf:.2%})"

        return modality_str, disease_dict
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(error_msg) # Log full error to console for debugging
        # Display the error message in the modality textbox for the user to see.
        return f"An error occurred: {e}", {"Error": 0.0}


# --- Gradio User Interface ---
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="orange")) as iface:
        gr.Markdown(
            """
            # 🩺 Multi-Modal Medical Image Analysis Pipeline
            Upload an X-ray, MRI, or CT scan. The system will first identify the modality, then route it to a specialized AI model for disease classification.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="📁 Upload Medical Image")
                btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
            with gr.Column(scale=2):
                output_modality = gr.Textbox(label="🔬 Detected Modality", interactive=False)
                output_disease = gr.Label(num_top_classes=3, label="Top 3 Predictions")

        btn.click(fn=predict, inputs=input_img, outputs=[output_modality, output_disease])
        input_img.change(fn=predict, inputs=input_img, outputs=[output_modality, output_disease])

        gr.Markdown("---")
        gr.Markdown("*Disclaimer: This tool is a demonstration and is not intended for clinical use. It is not a substitute for professional medical advice.*")

    return iface

# --- Launch the Application ---
if __name__ == "__main__":
    app = create_interface()
    app.launch(debug=True)
