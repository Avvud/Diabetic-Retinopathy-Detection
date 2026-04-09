# ─────────────────────────────────────────────────────────────────────────────
# app.py — DR Detection · Gradio UI
# Ensemble inference (EfficientNet-B0 + DenseNet121 + ResNet50)
# + Grad-CAM heatmap  +  Ollama llama3.2 clinical summary
#
# Run:  python app.py
# Prereqs:
#   pip install gradio torch torchvision pillow matplotlib requests
#   ollama pull llama3.2   (Ollama must be running: ollama serve)
# ─────────────────────────────────────────────────────────────────────────────

import os
import io
import json
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Gradio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    efficientnet_b3, densenet121, resnet50,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
NUM_CLASSES  = 5
IMG_SIZE     = 224
CLASS_NAMES  = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


WEIGHT_PATHS = {
    'efficientnet_b3': 'best_efficientnet_b3.pth',
    'densenet121':     'best_densenet121.pth',
    'resnet50':        'best_resnet50.pth',
}

GRADCAM_LAYERS = {
    'efficientnet_b3': 'features.8',
    'densenet121':     'features.denseblock4',
    'resnet50':        'layer4',
}

# Severity metadata used in UI badges and LLM prompt
SEVERITY_META = {
    'No_DR':          {'level': 0, 'color': '#2e7d32', 'label': 'No Diabetic Retinopathy'},
    'Mild':           {'level': 1, 'color': '#f57f17', 'label': 'Mild DR'},
    'Moderate':       {'level': 2, 'color': '#e65100', 'label': 'Moderate DR'},
    'Severe':         {'level': 3, 'color': '#b71c1c', 'label': 'Severe DR'},
    'Proliferate_DR': {'level': 4, 'color': '#4a148c', 'label': 'Proliferative DR'},
}

# ─────────────────────────────────────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────────────────────────────────────
def build_model(name: str) -> nn.Module:
    if name == 'efficientnet_b3':
        model = efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    elif name == 'densenet121':
        model = densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    elif name == 'resnet50':
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        raise ValueError(f'Unknown model: {name}')
    return model


def load_models() -> dict:
    models = {}
    for name, path in WEIGHT_PATHS.items():
        m = build_model(name).to(DEVICE)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.eval()
            models[name] = m
            print(f'  [OK] {path}')
        else:
            print(f'  [SKIP] {path} not found')
    return models


inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print(f"Device: {DEVICE}")
print("Loading models...")
MODELS = load_models()
print(f"Ready — {len(MODELS)} model(s) loaded.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(img_pil: Image.Image):
    img_tensor = inference_transform(img_pil).unsqueeze(0).to(DEVICE)
    softmax    = nn.Softmax(dim=1)
    prob_sum   = None

    for m in MODELS.values():
        probs    = softmax(m(img_tensor))
        prob_sum = probs if prob_sum is None else prob_sum + probs

    avg_probs  = prob_sum / len(MODELS)
    pred_idx   = avg_probs.argmax(dim=1).item()
    confidence = avg_probs[0][pred_idx].item()
    all_probs  = {CLASS_NAMES[i]: round(avg_probs[0][i].item() * 100, 2)
                  for i in range(NUM_CLASSES)}

    return CLASS_NAMES[pred_idx], pred_idx, confidence, img_tensor, all_probs

# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
def get_gradcam(model, model_name: str, img_tensor: torch.Tensor,
                target_class: int) -> np.ndarray:
    target_layer = dict(model.named_modules())[GRADCAM_LAYERS[model_name]]
    feat_map     = {}

    def fwd_hook(m, inp, out):
        feat_map['act'] = out
        out.retain_grad()

    handle = target_layer.register_forward_hook(fwd_hook)
    img_in = img_tensor.clone()
    with torch.enable_grad():
        logits = model(img_in)
        model.zero_grad()
        logits[0, target_class].backward()
    handle.remove()

    act  = feat_map.get('act')
    grad = act.grad if act is not None else None
    if act is None or grad is None or act.dim() != 4:
        return np.zeros((IMG_SIZE, IMG_SIZE))

    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam     = F.relu((weights * act).sum(dim=1).squeeze())
    cam_np  = cam.detach().cpu().numpy()

    if cam_np.max() > cam_np.min():
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
    else:
        cam_np = np.zeros_like(cam_np)

    cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
    return np.array(cam_pil) / 255.0


def ensemble_gradcam(img_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    return np.mean([
        get_gradcam(m, name, img_tensor, target_class)
        for name, m in MODELS.items()
    ], axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_heatmap_figure(img_pil: Image.Image, cam: np.ndarray,
                        label: str, confidence: float,
                        all_probs: dict) -> Image.Image:
    """Returns a PIL image of the 3-panel Grad-CAM figure."""
    img_np      = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
    heatmap_rgb = cm.jet(cam)[:, :, :3]
    overlay     = np.clip(0.45 * (img_np / 255.0) + 0.55 * heatmap_rgb, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor('#111827')

    # Panel 1 — original
    axes[0].imshow(img_np)
    axes[0].set_title('Original Scan', color='white', fontsize=11, pad=8)
    axes[0].axis('off')

    # Panel 2 — Grad-CAM
    axes[1].imshow(overlay)
    axes[1].set_title(
        f'Grad-CAM  ·  {SEVERITY_META[label]["label"]}  ({confidence*100:.1f}%)',
        color='#34d399', fontsize=11, fontweight='bold', pad=8
    )
    axes[1].axis('off')
    sm   = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Activation', color='#9ca3af', fontsize=8)
    cbar.ax.yaxis.set_tick_params(color='#9ca3af')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#9ca3af')

    plt.suptitle(
        'Diabetic Retinopathy Detection  ·  '
        'Ensemble: EfficientNet-B3 + DenseNet121 + ResNet50',
        color='#6b7280', fontsize=9, y=1.01
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

# ─────────────────────────────────────────────────────────────────────────────
# Ollama LLM summary
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(label: str, confidence: float,
                 all_probs: dict, patient_history: str) -> str:
    probs_str = "\n".join(
        f"  - {SEVERITY_META[c]['label']}: {p:.1f}%"
        for c, p in sorted(all_probs.items(),
                           key=lambda x: SEVERITY_META[x[0]]['level'])
    )
    history_section = (
        f"Patient History provided by clinician:\n{patient_history.strip()}"
        if patient_history and patient_history.strip()
        else "No patient history provided."
    )
    return f"""You are an experienced ophthalmologist assistant specializing in diabetic retinopathy (DR).

A deep learning ensemble model has analysed a retinal fundus image and returned the following results:

Classification Result:
  - Predicted class: {SEVERITY_META[label]['label']}
  - Confidence: {confidence*100:.1f}%

All class probabilities:
{probs_str}

{history_section}

Please provide a structured clinical summary with the following sections:

1. FINDINGS
   Briefly explain what the predicted DR severity level means clinically, and what retinal changes are typically present at this stage.

2. PATIENT RISK ASSESSMENT
   Based on the classification AND the patient history above, assess the patient's overall risk. Consider diabetes duration, glycaemic control, hypertension, or any other relevant factors mentioned.

3. CLINICAL RECOMMENDATIONS
   Provide clear, actionable next steps — e.g. referral urgency, follow-up interval, treatment considerations, lifestyle advice.

4. IMPORTANT DISCLAIMER
   Remind the clinician that this is an AI-assisted tool and the final diagnosis must be made by a qualified ophthalmologist.

Write in clear, professional clinical language. Use concise paragraphs, not bullet lists within sections.
"""


def query_ollama(prompt: str):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_ctx": 1024
                }
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                yield data.get("response", "")

    except Exception as e:
        yield f"⚠️ Ollama error: {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline (called by Gradio)
# ─────────────────────────────────────────────────────────────────────────────
def analyse(image: Image.Image, patient_history: str):
    if image is None:
        yield None, "⚠️  Please upload a retinal fundus image.", ""
        return

    if not MODELS:
        yield None, "⚠️  No model weights found. Check .pth files are in the working directory.", ""
        return

    # 1. Inference
    label, pred_idx, confidence, img_tensor, all_probs = run_inference(image)

    # 2. Grad-CAM
    cam = ensemble_gradcam(img_tensor, pred_idx)

    # 3. Heatmap figure
    heatmap_img = make_heatmap_figure(image, cam, label, confidence, all_probs)

    # 4. Severity badge markdown
    meta = SEVERITY_META[label]
    badge_md = (
        f"## Prediction:  {meta['label']}\n\n"
        f"**Confidence:** {confidence*100:.2f}%  |  "
        f"**Severity Level:** {meta['level']} / 4  |  "
        f"**Models:** {', '.join(MODELS.keys())}"
    )

    # Yield intermediate results to show images immediately
    yield heatmap_img, badge_md, "Generating clinical summary..."

    # Free PyTorch residual memory to maximize VRAM for Ollama
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. LLM summary
    prompt  = build_prompt(label, confidence, all_probs, patient_history)
    
    summary = ""
    for chunk in query_ollama(prompt):
        summary += chunk
        # Yield updated summary text step-by-step
        yield heatmap_img, badge_md, summary


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
THEME = gr.themes.Base(
    primary_hue="emerald",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    block_border_color="#334155",
    block_border_color_dark="#334155",
    block_label_text_color="#94a3b8",
    block_label_text_color_dark="#94a3b8",
    block_title_text_color="#f1f5f9",
    block_title_text_color_dark="#f1f5f9",
    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    input_border_color="#334155",
    input_border_color_dark="#334155",
    input_placeholder_color="#475569",
    button_primary_background_fill="#059669",
    button_primary_background_fill_hover="#047857",
    button_primary_text_color="white",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
)

with gr.Blocks(title="DR Detection") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding: 28px 0 12px;">
        <div style="font-size:13px; letter-spacing:3px; text-transform:uppercase;
                    color:#34d399; font-weight:600; margin-bottom:8px;">
            AI-Assisted Ophthalmology
        </div>
        <h1 style="font-size:2rem; font-weight:800; color:#f1f5f9; margin:0;">
            Diabetic Retinopathy Detection
        </h1>
        <p style="color:#64748b; margin-top:10px; font-size:0.95rem;">
            Ensemble model (EfficientNet-B3 · DenseNet121 · ResNet50) &nbsp;+&nbsp;
            Grad-CAM visualisation &nbsp;+&nbsp; LLaMA 3.2 clinical summary
        </p>
    </div>
    """)

    # ── Main layout ───────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # Left column — inputs
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 🩺 Patient Input")

            image_input = gr.Image(
                type="pil",
                label="Retinal Fundus Image",
                image_mode="RGB",
                height=280,
                sources=["upload", "clipboard"],
            )

            history_input = gr.Textbox(
                label="Patient History",
                placeholder=(
                    "e.g. 58-year-old male, Type 2 diabetes for 12 years, "
                    "HbA1c 9.2%, hypertensive on amlodipine, no prior laser treatment..."
                ),
                lines=7,
                max_lines=14,
            )

            analyse_btn = gr.Button(
                "🔬  Analyse",
                variant="primary",
                size="lg",
            )

            gr.HTML("""
            <div style="margin-top:12px; padding:12px; border-radius:8px;
                        background:#1e3a5f; border:1px solid #1d4ed8; font-size:0.8rem;
                        color:#93c5fd; line-height:1.6;">
                <strong>ℹ️  Note:</strong> Ensure Ollama is running and
                <code>llama3.2</code> is pulled before clicking Analyse.
                The heatmap highlights regions the model focused on — red/yellow
                = high activation.
            </div>
            """)

        # Right column — outputs
        with gr.Column(scale=2, min_width=500):
            gr.Markdown("### 📊 Analysis Results")

            result_badge = gr.Markdown(
                value="_Upload an image and click Analyse to see results._"
            )

            heatmap_output = gr.Image(
                label="Grad-CAM Heatmap · Original",
                type="pil",
                height=320,
                interactive=False,
            )

            gr.Markdown("### 🧠 Clinical Summary  *(LLaMA 3.2)*")

            summary_output = gr.Textbox(
                label="",
                lines=18,
                max_lines=40,
                interactive=False,
                
                placeholder="Clinical summary will appear here after analysis...",
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding:20px 0 8px;
                color:#475569; font-size:0.78rem; border-top:1px solid #1e293b; margin-top:20px;">
        ⚠️ This tool is for <strong>research and clinical decision support only</strong>.
        It does not replace professional ophthalmological diagnosis.
    </div>
    """)

    # ── Wire up ───────────────────────────────────────────────────────────────
    analyse_btn.click(
        fn=analyse,
        inputs=[image_input, history_input],
        outputs=[heatmap_output, result_badge, summary_output],
        show_progress="full",
    )

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=True,
        inbrowser=True,
        theme=THEME,
    )