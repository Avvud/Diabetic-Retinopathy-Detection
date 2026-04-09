# ─────────────────────────────────────────────────────────────────────────────
# inference.py — Diabetic Retinopathy Detection (No Training Required)
# Loads saved .pth weights, runs ensemble inference, and produces a
# Grad-CAM heatmap overlay showing which regions drove the prediction.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    densenet121, DenseNet121_Weights,
    resnet50, ResNet50_Weights,
)

# ── Config ────────────────────────────────────────────────────────────────────
NUM_CLASSES = 5
IMG_SIZE    = 224
CLASS_NAMES = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WEIGHT_PATHS = {
    'efficientnet_b3': 'best_efficientnet_b3.pth',
    'densenet121':     'best_densenet121.pth',
    'resnet50':        'best_resnet50.pth',
}

# Sub-layer to hook for each architecture.
# Chosen specifically to sit BEFORE any inplace ops on the output tensor.
GRADCAM_LAYERS = {
    'efficientnet_b3': 'features.8',             # Conv2dNormActivation, last feature block
    'densenet121':     'features.denseblock4',   # last dense block, before inplace relu
    'resnet50':        'layer4',                 # last residual block
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Model Factory ─────────────────────────────────────────────────────────────
def build_model(name: str) -> nn.Module:
    if name == 'efficientnet_b3':
        model = efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    elif name == 'densenet121':
        model = densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES)
    elif name == 'resnet50':
        model = resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
    else:
        raise ValueError(f'Unknown model: {name}')
    return model

# ── Load Weights ──────────────────────────────────────────────────────────────
def load_models(weight_paths: dict, device) -> dict:
    models = {}
    for name, path in weight_paths.items():
        model = build_model(name).to(device)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models[name] = model
            print(f'  [OK] Loaded {path}')
        else:
            print(f'  [SKIP] {path} not found — excluded from ensemble')
    return models

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def get_gradcam(model, model_name: str, img_tensor: torch.Tensor,
                target_class: int) -> np.ndarray:
    """
    Grad-CAM using retain_grad() on the feature activation.
    Avoids register_full_backward_hook entirely — sidestepping the
    inplace-ReLU conflict in DenseNet121.
    """
    # Find the target layer
    target_layer = dict(model.named_modules())[GRADCAM_LAYERS[model_name]]

    feat_map = {}

    # Forward hook: capture output and call retain_grad() so we can
    # read .grad on it after backward (works even for non-leaf tensors)
    def fwd_hook(m, inp, out):
        # Detach from any inplace ops downstream by using out directly;
        # retain_grad lets us access gradient on this non-leaf tensor
        feat_map['act'] = out
        out.retain_grad()

    handle = target_layer.register_forward_hook(fwd_hook)

    # Run forward pass with grad enabled
    img_in = img_tensor.clone()
    with torch.enable_grad():
        logits = model(img_in)
        model.zero_grad()
        # Scalar backward on the target class score
        score = logits[0, target_class]
        score.backward()

    handle.remove()

    act  = feat_map['act']            # (1, C, H, W)
    grad = act.grad                   # (1, C, H, W)

    if act is None or grad is None or act.dim() != 4:
        return np.zeros((IMG_SIZE, IMG_SIZE))

    # Global average pool gradients → per-channel weights
    weights = grad.mean(dim=[2, 3], keepdim=True)     # (1, C, 1, 1)
    cam     = (weights * act).sum(dim=1).squeeze()    # (H, W)
    cam     = F.relu(cam)                             # keep only positive

    cam_np = cam.detach().cpu().numpy()
    if cam_np.max() > cam_np.min():
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
    else:
        cam_np = np.zeros_like(cam_np)

    # Resize to input image size
    cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
    return np.array(cam_pil) / 255.0


def ensemble_gradcam(models: dict, img_tensor: torch.Tensor,
                     target_class: int) -> np.ndarray:
    """Average Grad-CAM maps across all ensemble models."""
    maps = []
    for name, model in models.items():
        cam = get_gradcam(model, name, img_tensor, target_class)
        maps.append(cam)
    return np.mean(maps, axis=0)

# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(image_path: str, models: dict, transform, device):
    """Ensemble softmax inference."""
    img_pil    = Image.open(image_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    softmax  = nn.Softmax(dim=1)
    prob_sum = None

    for model in models.values():
        probs    = softmax(model(img_tensor))
        prob_sum = probs if prob_sum is None else prob_sum + probs

    avg_probs  = prob_sum / len(models)
    pred_idx   = avg_probs.argmax(dim=1).item()
    confidence = avg_probs[0][pred_idx].item()

    all_probs = {CLASS_NAMES[i]: round(avg_probs[0][i].item() * 100, 2)
                 for i in range(NUM_CLASSES)}

    return CLASS_NAMES[pred_idx], pred_idx, confidence, img_pil, img_tensor, all_probs

# ── Visualisation ─────────────────────────────────────────────────────────────
def save_result(img_pil: Image.Image, cam: np.ndarray,
                label: str, confidence: float,
                all_probs: dict, out_path: str = 'prediction_result.png',
                show_plot: bool = True):
    """
    3-panel figure:
      Left   — original image
      Centre — Grad-CAM heatmap overlay
      Right  — class probability bar chart
    """
    img_np = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))

    # Jet colourmap over heatmap, blended with original
    heatmap_rgb = cm.jet(cam)[:, :, :3]
    overlay     = 0.45 * (img_np / 255.0) + 0.55 * heatmap_rgb
    overlay     = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0f0f0f')

    # ── Panel 1: Original ────────────────────────────────────────────────────
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', color='white', fontsize=12, pad=10)
    axes[0].axis('off')

    # ── Panel 2: Grad-CAM overlay ─────────────────────────────────────────────
    axes[1].imshow(overlay)
    axes[1].set_title(
        f'Grad-CAM  ·  {label}  ({confidence*100:.1f}%)',
        color='#00e676', fontsize=12, fontweight='bold', pad=10
    )
    axes[1].axis('off')

    sm   = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Activation intensity', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # ── Panel 3: Probability bars ─────────────────────────────────────────────
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1])
    classes = [c for c, _ in sorted_probs]
    values  = [v for _, v in sorted_probs]

    bar_colors = ['#ef5350' if c == label else '#42a5f5' for c in classes]
    bars = axes[2].barh(classes, values, color=bar_colors,
                        edgecolor='white', linewidth=0.5, height=0.6)
    for bar, val in zip(bars, values):
        axes[2].text(
            min(val + 1.0, 98), bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', color='white',
            fontsize=10, fontweight='bold'
        )

    axes[2].set_xlim(0, 110)
    axes[2].set_xlabel('Confidence (%)', color='white', fontsize=10)
    axes[2].set_title('Class Probabilities', color='white', fontsize=12, pad=10)
    axes[2].set_facecolor('#1a1a2e')
    axes[2].tick_params(colors='white')
    axes[2].spines[:].set_color('#444')
    for lbl in axes[2].get_yticklabels():
        lbl.set_color('white')

    plt.suptitle(
        'Diabetic Retinopathy Detection  ·  '
        'Ensemble (EfficientNet-B3 + DenseNet121 + ResNet50)',
        color='white', fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    if show_plot:
        plt.show()
    plt.close(fig)
    print(f'\nResult saved → {out_path}')

# ── Main ──────────────────────────────────────────────────────────────────────
def process_single_image(image_path, models, device, out_path_prefix='', show_plot=True):
    print(f'\nRunning inference on: {image_path}')
    label, pred_idx, confidence, img_pil, img_tensor, all_probs = predict(
        image_path, models, inference_transform, device
    )

    # ── Print results ──────────────────────────────────────────────────────────
    print('\n' + '='*45)
    print(f'  Prediction  : {label}')
    print(f'  Confidence  : {confidence*100:.2f}%')
    print(f'  Models used : {", ".join(models.keys())}')
    print('='*45)
    print('\nAll class probabilities:')
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar = '█' * int(prob / 2)
        print(f'  {cls:<18} {prob:6.2f}%  {bar}')

    # ── Grad-CAM ───────────────────────────────────────────────────────────────
    print('\nGenerating Grad-CAM heatmap...')
    cam = ensemble_gradcam(models, img_tensor, pred_idx)

    # ── Save final figure ──────────────────────────────────────────────────────
    if out_path_prefix:
        os.makedirs(out_path_prefix, exist_ok=True)
        filename = os.path.basename(image_path)
        base, ext = os.path.splitext(filename)
        out_path = os.path.join(out_path_prefix, f"{base}_result.png")
    else:
        out_path = 'prediction_result.png'

    save_result(img_pil, cam, label, confidence, all_probs, out_path=out_path, show_plot=show_plot)
    return label, out_path


def launch_gradio_interface(models, device):
    import gradio as gr
    
    def process_folder(folder_path, uploaded_files):
        image_files = []
        
        # Check Local Folder
        if folder_path and os.path.exists(folder_path):
            if os.path.isdir(folder_path):
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            image_files.append(os.path.join(root, file))
            elif os.path.isfile(folder_path):
                if folder_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_files.append(folder_path)

        # Check Uploaded Folder/Files
        if uploaded_files:
            for f in uploaded_files:
                path = f.name if hasattr(f, 'name') else f
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_files.append(path)
                    
        # Remove duplicates
        image_files = list(set(image_files))

        if not image_files:
            return None, "No valid images found in the specified path or upload."

        out_folder = 'gradio_inference_results'
        os.makedirs(out_folder, exist_ok=True)
        
        gallery_items = []
        summary_counts = {c: 0 for c in CLASS_NAMES}
        
        for img_path in image_files:
            label, res_path = process_single_image(img_path, models, device, out_path_prefix=out_folder, show_plot=False)
            gallery_items.append((res_path, f"{os.path.basename(img_path)} ({label})"))
            summary_counts[label] += 1
            
        summary_md = "### Analysis Summary\n"
        summary_md += f"**Total Images Processed**: {len(image_files)}\n\n"
        for c in CLASS_NAMES:
            summary_md += f"- **{c}**: {summary_counts[c]}\n"
            
        return gallery_items, summary_md

    with gr.Blocks(title="Diabetic Retinopathy Analyzer") as demo:
        gr.Markdown("# Diabetic Retinopathy Batch Analyzer\nProvide a local folder path or upload a folder of retinal images for ensemble Grad-CAM inference.")
        
        with gr.Row():
            with gr.Column():
                folder_input = gr.Textbox(label="Local Folder Path (Optional)", placeholder="e.g., B:\\path\\to\\images")
                file_upload = gr.File(label="Upload Folder / Multiple Images", file_count="directory", type="filepath")
                analyze_btn = gr.Button("Analyze Images", variant="primary")
                
            with gr.Column():
                summary_output = gr.Markdown("Analysis summary will appear here.")
                
        gallery_output = gr.Gallery(label="Inference Results", columns=2, object_fit="contain", height="auto")
        
        analyze_btn.click(
            fn=process_folder,
            inputs=[folder_input, file_upload],
            outputs=[gallery_output, summary_output]
        )
        
    demo.launch(inbrowser=True, theme=gr.themes.Soft())


if __name__ == '__main__':
    print(f'\nDevice : {DEVICE}')
    print('Loading models...')
    models = load_models(WEIGHT_PATHS, DEVICE)

    if not models:
        print('No models loaded. Check that .pth files are present.')
        sys.exit(1)

    if len(sys.argv) > 1:
        path_arg = sys.argv[1]

        if not os.path.exists(path_arg):
            print(f'Error: Path not found at "{path_arg}"')
            print('Usage: python inference.py [path/to/retinal_image.png_OR_folder]')
            sys.exit(1)
            
        if os.path.isdir(path_arg):
            # Batch processing
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            image_files = []
            for root, dirs, files in os.walk(path_arg):
                for file in files:
                    if file.lower().endswith(valid_exts):
                        image_files.append(os.path.join(root, file))

            print(f'\nFound {len(image_files)} images in directory "{path_arg}" for batch processing.')
            out_folder = 'inference_results'
            for img_path in image_files:
                process_single_image(img_path, models, DEVICE, out_path_prefix=out_folder, show_plot=False)
            print(f'\nBatch processing complete. Results saved in "{out_folder}/"')
        else:
            # Single image processing
            process_single_image(path_arg, models, DEVICE, show_plot=True)
    else:
        # Launch Gradio App
        print("\nStarting Gradio interface...")
        try:
            launch_gradio_interface(models, DEVICE)
        except ImportError:
            print("Please install Gradio to run the GUI: pip install gradio")