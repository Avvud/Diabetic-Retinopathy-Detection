# ─────────────────────────────────────────────────────────────────────────────
# app.py — DR Detection · Gradio UI
# Ensemble inference (EfficientNet-B0 + DenseNet121 + ResNet50)
# + Grad-CAM heatmap
# + SAM2/SAM segmentation (Grad-CAM → bounding box prompt → lesion mask)
# + Macular proximity weighting
# + Ollama llama3.2 clinical summary
#
# Run:  python app.py
#
# Prereqs:
#   pip install gradio torch torchvision pillow matplotlib requests opencv-python scipy
#
#   SAM2 (preferred):
#     pip install git+https://github.com/facebookresearch/sam2.git
#     # download checkpoint, e.g.:
#     # wget https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_small.pt
#
#   SAM fallback:
#     pip install git+https://github.com/facebookresearch/segment-anything.git
#     # wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
#
#   Ollama:
#     ollama serve  &&  ollama pull llama3.2
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import io
import warnings
import requests
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models import efficientnet_b0, densenet121, resnet50

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
    'efficientnet_b0': 'best_efficientnet_b0.pth',
    'densenet121':     'best_densenet121.pth',
    'resnet50':        'best_resnet50.pth',
}

GRADCAM_LAYERS = {
    'efficientnet_b0': 'features.8',
    'densenet121':     'features.denseblock4',
    'resnet50':        'layer4',
}

SEVERITY_META = {
    'No_DR':          {'level': 0, 'color': '#2e7d32', 'label': 'No Diabetic Retinopathy'},
    'Mild':           {'level': 1, 'color': '#f57f17', 'label': 'Mild DR'},
    'Moderate':       {'level': 2, 'color': '#e65100', 'label': 'Moderate DR'},
    'Severe':         {'level': 3, 'color': '#b71c1c', 'label': 'Severe DR'},
    'Proliferate_DR': {'level': 4, 'color': '#4a148c', 'label': 'Proliferative DR'},
}

# SAM checkpoint paths — checked in order; first found wins
SAM2_CHECKPOINTS = [
    'sam2_hiera_large.pt',
    'sam2_hiera_base_plus.pt',
    'sam2_hiera_small.pt',
    'sam2_hiera_tiny.pt',
]
SAM2_CONFIGS = {
    'sam2_hiera_large.pt':      'sam2_hiera_l.yaml',
    'sam2_hiera_base_plus.pt':  'sam2_hiera_b+.yaml',
    'sam2_hiera_small.pt':      'sam2_hiera_s.yaml',
    'sam2_hiera_tiny.pt':       'sam2_hiera_t.yaml',
}
SAM_CHECKPOINTS = [
    'sam_vit_h_4b8939.pth',
    'sam_vit_l_0b3195.pth',
    'sam_vit_b_01ec64.pth',
]
SAM_MODEL_TYPES = {
    'sam_vit_h_4b8939.pth': 'vit_h',
    'sam_vit_l_0b3195.pth': 'vit_l',
    'sam_vit_b_01ec64.pth': 'vit_b',
}

# Macular zone: central 25% of image radius is considered macula
MACULA_RADIUS_FRACTION = 0.25
# Per-class severity boost/penalty multipliers when lesion is central/peripheral
CENTRAL_BOOST    = 1.25   # lesion near macula → increase non-No_DR scores
PERIPHERAL_DAMP  = 0.85   # lesion peripheral → slightly reduce severity scores
# Grad-CAM threshold for SAM prompt extraction
GRADCAM_THRESHOLD    = 0.65   # fraction of max activation — high = tighter prompt
MIN_CONTOUR_AREA     = 200    # px² — ignore tiny noise blobs
MAX_BOX_IMAGE_FRAC   = 0.40   # reject box if it covers >40% of image area


# ─────────────────────────────────────────────────────────────────────────────
# ① SAM / SAM2 loader
# ─────────────────────────────────────────────────────────────────────────────
def load_sam_model():
    """
    Try SAM2 first (preferred), then fall back to original SAM.
    Returns (predictor, backend_name) or (None, 'none') if neither is available.
    """
    # ── Try SAM2 ──────────────────────────────────────────────────────────────
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        for ckpt in SAM2_CHECKPOINTS:
            if os.path.exists(ckpt):
                cfg = SAM2_CONFIGS[ckpt]
                print(f'  [SAM2] Loading {ckpt} with config {cfg}')
                sam2_model = build_sam2(cfg, ckpt, device=DEVICE)
                predictor  = SAM2ImagePredictor(sam2_model)
                print(f'  [SAM2] Ready.')
                return predictor, 'sam2'

        print('  [SAM2] Installed but no checkpoint found — trying SAM.')
    except ImportError:
        print('  [SAM2] Not installed — trying SAM.')

    # ── Try original SAM ──────────────────────────────────────────────────────
    try:
        from segment_anything import sam_model_registry, SamPredictor

        for ckpt in SAM_CHECKPOINTS:
            if os.path.exists(ckpt):
                mtype = SAM_MODEL_TYPES[ckpt]
                print(f'  [SAM]  Loading {ckpt} (type={mtype})')
                sam       = sam_model_registry[mtype](checkpoint=ckpt)
                sam.to(DEVICE)
                predictor = SamPredictor(sam)
                print(f'  [SAM]  Ready.')
                return predictor, 'sam'

        print('  [SAM]  Installed but no checkpoint found.')
    except ImportError:
        print('  [SAM]  Not installed.')

    print('  [SAM]  Segmentation disabled — results will use classifier output only.')
    return None, 'none'


# ─────────────────────────────────────────────────────────────────────────────
# ② Grad-CAM → SAM prompt bounding box
# ─────────────────────────────────────────────────────────────────────────────
def get_sam_prompt_from_gradcam(
    cam: np.ndarray,
    img_w: int,
    img_h: int,
    threshold: float = GRADCAM_THRESHOLD,
    min_area: int = MIN_CONTOUR_AREA,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Derive a tight SAM prompt from a Grad-CAM activation map.

    Strategy:
      1. Resize cam to image dimensions.
      2. Threshold at `threshold * max` (high threshold → tight region).
      3. Morphological closing to bridge small gaps.
      4. Find the largest contour above min_area.
      5. Compute bounding box and centroid point.
      6. Guard: if the box covers > MAX_BOX_IMAGE_FRAC of the image,
         Grad-CAM is too diffuse to be a useful box prompt — return
         None for the box so the caller can switch to point-only mode.

    Returns:
        box      : np.ndarray (4,) [x0,y0,x1,y1] or None if too large / not found
        centroid : np.ndarray (2,) [cx, cy] pixel coords of activation centroid, or None
        mask     : np.ndarray bool (H,W) binary Grad-CAM mask, or None
    """
    cam_resized = cv2.resize(cam.astype(np.float32), (img_w, img_h),
                             interpolation=cv2.INTER_LINEAR)

    thresh_val = threshold * cam_resized.max()
    binary     = (cam_resized >= thresh_val).astype(np.uint8) * 255

    # Morphological closing — fill small gaps but keep it tight
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None, None, None

    x, y, w, h = cv2.boundingRect(largest)

    # Centroid of the largest contour (used as point prompt fallback)
    M = cv2.moments(largest)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2
    centroid = np.array([cx, cy], dtype=np.float32)

    # ── Box size guard ────────────────────────────────────────────────────────
    box_frac = (w * h) / (img_w * img_h)
    if box_frac > MAX_BOX_IMAGE_FRAC:
        # Box is too large — Grad-CAM is diffuse.
        # Return centroid only; caller will use point prompt instead.
        return None, centroid, (closed > 0)

    # Small margin and clamp
    margin_x = max(int(w * 0.05), 4)
    margin_y = max(int(h * 0.05), 4)
    x0 = max(0,     x - margin_x)
    y0 = max(0,     y - margin_y)
    x1 = min(img_w, x + w + margin_x)
    y1 = min(img_h, y + h + margin_y)

    box  = np.array([x0, y0, x1, y1], dtype=np.float32)
    mask = (closed > 0)

    return box, centroid, mask


# ─────────────────────────────────────────────────────────────────────────────
# ③ SAM segmentation
# ─────────────────────────────────────────────────────────────────────────────
def run_sam_segmentation(
    predictor,
    backend: str,
    img_np: np.ndarray,
    box: np.ndarray | None,
    centroid: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Run SAM/SAM2 with either a bounding-box prompt or a centroid point prompt.

    Priority:
      • box is valid → use box prompt (most precise)
      • box is None but centroid is valid → use point prompt (Grad-CAM too diffuse)
      • both None → return None

    After segmentation, rejects masks that cover >50% of the image
    (indicates SAM segmented the whole fundus, not a lesion).

    Args:
        predictor : SAM or SAM2 predictor object
        backend   : 'sam2' | 'sam'
        img_np    : uint8 RGB numpy array (H, W, 3)
        box       : [x0, y0, x1, y1] or None
        centroid  : [cx, cy] pixel coords or None (point prompt fallback)

    Returns:
        best_mask : bool np.ndarray (H, W), or None on failure
    """
    if predictor is None:
        return None
    if box is None and centroid is None:
        return None

    img_h, img_w = img_np.shape[:2]
    MAX_MASK_COVERAGE = 0.50   # reject mask if it covers >50% of image

    def _pick_best(masks, scores):
        """Return highest-scoring mask that is not a whole-image segment."""
        if masks is None or len(masks) == 0:
            return None
        order = np.argsort(scores)[::-1]
        for idx in order:
            m = masks[idx].astype(bool)
            coverage = m.sum() / m.size
            if 0.001 <= coverage <= MAX_MASK_COVERAGE:
                return m
        return None

    try:
        predictor.set_image(img_np)

        if box is not None:
            # ── Box prompt ────────────────────────────────────────────────────
            if backend == 'sam2':
                masks, scores, _ = predictor.predict(
                    box=box[None], multimask_output=True)
            else:
                masks, scores, _ = predictor.predict(
                    box=box, multimask_output=True)

            best = _pick_best(masks, scores)
            if best is not None:
                return best
            # Box prompt produced a bad mask → fall through to point prompt

        if centroid is not None:
            # ── Point prompt fallback ─────────────────────────────────────────
            pt = centroid.astype(int)
            point_coords  = np.array([[pt[0], pt[1]]])
            point_labels  = np.array([1])          # 1 = foreground

            if backend == 'sam2':
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
            else:
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )

            return _pick_best(masks, scores)

        return None

    except Exception as exc:
        warnings.warn(f'SAM segmentation failed: {exc}')
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ④ Macular proximity
# ─────────────────────────────────────────────────────────────────────────────
def compute_central_retina_weight(
    mask: np.ndarray,
    img_h: int,
    img_w: int,
    macula_radius_frac: float = MACULA_RADIUS_FRACTION,
) -> tuple[float, str]:
    """
    Estimate what fraction of the segmented lesion mask falls inside the
    macular zone (central circle of radius = macula_radius_frac * min(H,W)/2).

    Returns:
        central_fraction : float in [0, 1]
        zone_label       : human-readable string
    """
    if mask is None or mask.sum() == 0:
        return 0.0, 'unknown'

    cy, cx     = img_h / 2, img_w / 2
    mac_radius = macula_radius_frac * min(img_h, img_w) / 2

    # Build distance map from image centre
    ys, xs    = np.mgrid[0:img_h, 0:img_w]
    dist_map  = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    macula_zone = dist_map <= mac_radius

    lesion_pixels   = mask.sum()
    central_pixels  = (mask & macula_zone).sum()
    central_frac    = central_pixels / lesion_pixels if lesion_pixels > 0 else 0.0

    if central_frac >= 0.5:
        zone_label = 'central / macular'
    elif central_frac >= 0.20:
        zone_label = 'para-central'
    else:
        zone_label = 'peripheral'

    return float(central_frac), zone_label


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ Probability adjustment
# ─────────────────────────────────────────────────────────────────────────────
def adjust_probabilities_with_segmentation(
    all_probs: dict[str, float],
    central_frac: float,
    zone_label: str,
    sam_succeeded: bool,
) -> tuple[dict[str, float], str]:
    """
    Apply macular-proximity weighting to the raw classifier probabilities.

    Rules:
      • Central (≥50% of lesion in macula): boost all non-No_DR classes by
        CENTRAL_BOOST, renormalise.
      • Peripheral (<20% central): damp non-No_DR classes by PERIPHERAL_DAMP,
        renormalise.
      • Para-central or SAM failed: no adjustment.

    Returns:
        adjusted_probs : dict class→float (sum ≈ 100)
        adjustment_note: human-readable explanation
    """
    if not sam_succeeded or zone_label == 'unknown':
        return all_probs, 'No SAM mask — original classifier output used.'

    probs_arr = np.array([all_probs[c] for c in CLASS_NAMES], dtype=np.float64)
    no_dr_idx = CLASS_NAMES.index('No_DR')

    if zone_label == 'central / macular':
        # Boost severity classes
        for i, c in enumerate(CLASS_NAMES):
            if c != 'No_DR':
                probs_arr[i] *= CENTRAL_BOOST
        note = (f'⚠️  Lesion is {zone_label} ({central_frac*100:.0f}% macular overlap). '
                f'Severity scores boosted ×{CENTRAL_BOOST}.')

    elif zone_label == 'peripheral':
        # Damp severity classes
        for i, c in enumerate(CLASS_NAMES):
            if c != 'No_DR':
                probs_arr[i] *= PERIPHERAL_DAMP
        note = (f'ℹ️  Lesion is {zone_label} ({central_frac*100:.0f}% macular overlap). '
                f'Severity scores dampened ×{PERIPHERAL_DAMP}.')

    else:  # para-central
        note = (f'ℹ️  Lesion is {zone_label} ({central_frac*100:.0f}% macular overlap). '
                f'No probability adjustment applied.')
        return all_probs, note

    # Renormalise to sum=100
    total = probs_arr.sum()
    if total > 0:
        probs_arr = probs_arr / total * 100.0

    adjusted = {CLASS_NAMES[i]: round(float(probs_arr[i]), 2)
                for i in range(NUM_CLASSES)}
    return adjusted, note


# ─────────────────────────────────────────────────────────────────────────────
# Classifier models
# ─────────────────────────────────────────────────────────────────────────────
def build_model(name: str) -> nn.Module:
    if name == 'efficientnet_b0':
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
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
print("Loading classifier models...")
MODELS = load_models()
print(f"Classifier ready — {len(MODELS)} model(s).\n")

print("Loading SAM/SAM2...")
SAM_PREDICTOR, SAM_BACKEND = load_sam_model()
print()


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
def _prob_bars(ax, all_probs: dict, highlight: str):
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1])
    classes = [SEVERITY_META[c]['label'] for c, _ in sorted_probs]
    values  = [v for _, v in sorted_probs]
    raw_cls = [c for c, _ in sorted_probs]
    colors  = [SEVERITY_META[c]['color'] for c in raw_cls]

    bars = ax.barh(classes, values, color=colors,
                   edgecolor='#374151', linewidth=0.6, height=0.55)
    for bar, val in zip(bars, values):
        ax.text(min(val + 1, 97), bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', color='white',
                fontsize=8.5, fontweight='bold')

    ax.set_xlim(0, 112)
    ax.set_xlabel('Confidence (%)', color='#9ca3af', fontsize=8)
    ax.set_facecolor('#1f2937')
    ax.tick_params(colors='#9ca3af')
    ax.spines[:].set_color('#374151')
    for lbl in ax.get_yticklabels():
        lbl.set_color('white')
        lbl.set_fontsize(8)


def make_heatmap_figure(img_pil: Image.Image, cam: np.ndarray,
                        label: str, confidence: float,
                        all_probs: dict) -> Image.Image:
    """3-panel: original | Grad-CAM | probabilities."""
    img_np      = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
    heatmap_rgb = cm.jet(cam)[:, :, :3]
    overlay     = np.clip(0.45 * (img_np / 255.0) + 0.55 * heatmap_rgb, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor('#111827')

    axes[0].imshow(img_np)
    axes[0].set_title('Original Scan', color='white', fontsize=11, pad=8)
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title(
        f'Grad-CAM  ·  {SEVERITY_META[label]["label"]}  ({confidence*100:.1f}%)',
        color='#34d399', fontsize=11, fontweight='bold', pad=8)
    axes[1].axis('off')
    sm   = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Activation', color='#9ca3af', fontsize=8)
    cbar.ax.yaxis.set_tick_params(color='#9ca3af')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#9ca3af')

    axes[2].set_title('Class Probabilities', color='white', fontsize=11, pad=8)
    _prob_bars(axes[2], all_probs, label)

    plt.suptitle('Grad-CAM · Ensemble: EfficientNet-B0 + DenseNet121 + ResNet50',
                 color='#6b7280', fontsize=8.5, y=1.01)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_sam_figure(
    img_pil: Image.Image,
    mask: np.ndarray | None,
    box: np.ndarray | None,
    adj_probs: dict,
    orig_label: str,
    adj_label: str,
    zone_label: str,
    sam_note: str,
) -> Image.Image:
    """
    3-panel SAM figure:
      Left  — original with SAM bounding-box prompt
      Centre — mask overlay
      Right  — adjusted class probabilities
    """
    img_np  = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor('#111827')

    # Panel 1 — original + bounding box
    axes[0].imshow(img_np)
    if box is not None:
        # Scale box from original image space to IMG_SIZE
        orig_w, orig_h = img_pil.size
        sx, sy = IMG_SIZE / orig_w, IMG_SIZE / orig_h
        bx0, by0 = box[0] * sx, box[1] * sy
        bx1, by1 = box[2] * sx, box[3] * sy
        rect = plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                              linewidth=2, edgecolor='#facc15',
                              facecolor='none', linestyle='--')
        axes[0].add_patch(rect)
        axes[0].text(bx0, max(by0 - 5, 0), 'SAM prompt',
                     color='#facc15', fontsize=7, fontweight='bold')
    axes[0].set_title('SAM Bounding Box Prompt', color='white', fontsize=11, pad=8)
    axes[0].axis('off')

    # Panel 2 — mask overlay
    if mask is not None:
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        overlay = img_np.copy().astype(np.float32) / 255.0
        overlay[mask_resized] = overlay[mask_resized] * 0.4 + np.array([1.0, 0.2, 0.2]) * 0.6
        overlay = np.clip(overlay, 0, 1)
        axes[1].imshow(overlay)

        # Draw macula circle
        cy, cx = IMG_SIZE // 2, IMG_SIZE // 2
        mac_r  = MACULA_RADIUS_FRACTION * IMG_SIZE / 2
        circ   = plt.Circle((cx, cy), mac_r, color='#60a5fa',
                             fill=False, linewidth=1.5, linestyle=':')
        axes[1].add_patch(circ)
        axes[1].text(cx, cy - mac_r - 5, 'macula zone',
                     color='#60a5fa', fontsize=7, ha='center')

        changed = '→ ' + SEVERITY_META[adj_label]['label'] if adj_label != orig_label else ''
        axes[1].set_title(
            f'SAM Mask  ·  {zone_label}  {changed}',
            color='#f472b6', fontsize=10, fontweight='bold', pad=8)
    else:
        axes[1].imshow(img_np)
        axes[1].set_title('SAM: no reliable mask found', color='#9ca3af',
                          fontsize=10, pad=8)
    axes[1].axis('off')

    # Panel 3 — adjusted probabilities
    axes[2].set_title('Adjusted Probabilities', color='white', fontsize=11, pad=8)
    _prob_bars(axes[2], adj_probs, adj_label)

    # Annotation note
    fig.text(0.5, -0.03, sam_note, ha='center', color='#94a3b8',
             fontsize=8, wrap=True)

    plt.suptitle('SAM Segmentation · Macular Proximity Weighting',
                 color='#6b7280', fontsize=8.5, y=1.01)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ─────────────────────────────────────────────────────────────────────────────
# Ollama
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(label: str, confidence: float, all_probs: dict,
                 patient_history: str, adj_label: str,
                 zone_label: str, sam_note: str) -> str:
    probs_str = "\n".join(
        f"  - {SEVERITY_META[c]['label']}: {p:.1f}%"
        for c, p in sorted(all_probs.items(),
                           key=lambda x: SEVERITY_META[x[0]]['level'])
    )
    history_section = (
        f"Patient History:\n{patient_history.strip()}"
        if patient_history and patient_history.strip()
        else "No patient history provided."
    )
    sam_section = (
        f"SAM Segmentation:\n"
        f"  - Lesion zone: {zone_label}\n"
        f"  - Adjusted prediction: {SEVERITY_META[adj_label]['label']}\n"
        f"  - Note: {sam_note}"
        if adj_label else "SAM segmentation: not available."
    )

    return f"""You are an experienced ophthalmologist assistant specialising in diabetic retinopathy (DR).

A deep learning ensemble model analysed a retinal fundus image:

Classification:
  - Initial prediction : {SEVERITY_META[label]['label']} ({confidence*100:.1f}%)

Adjusted probabilities (after SAM macular weighting):
{probs_str}

{sam_section}

{history_section}

Please provide a structured clinical summary:

1. FINDINGS
   Explain what the predicted DR severity level means clinically and what retinal changes are typically present. Mention the SAM-identified lesion location (central vs peripheral) and its clinical significance.

2. PATIENT RISK ASSESSMENT
   Based on the classification, lesion location, and patient history, assess overall risk.

3. CLINICAL RECOMMENDATIONS
   Clear actionable next steps: referral urgency, follow-up interval, treatment considerations, lifestyle advice.

4. IMPORTANT DISCLAIMER
   Remind the clinician this is an AI-assisted tool; final diagnosis must come from a qualified ophthalmologist.

Write in clear, professional clinical language using concise paragraphs.
"""


def query_ollama(prompt: str):
    import json
    try:
        resp = requests.post(OLLAMA_URL,
                             json={"model": OLLAMA_MODEL, "prompt": prompt,
                                   "stream": True, "options": {"num_ctx": 1024}},
                             stream=True,
                             timeout=120)
        resp.raise_for_status()
        
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                yield data.get("response", "")

    except Exception as e:
        yield f"⚠️  Ollama error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def analyse(image: Image.Image, patient_history: str):
    """
    Full pipeline:
      1. Ensemble classifier  → label, confidence, raw probs
      2. Grad-CAM             → activation map
      3. SAM prompt           → bounding box from Grad-CAM
      4. SAM segmentation     → lesion mask
      5. Macular weighting    → adjusted probs, new label
      6. Visualisation        → Grad-CAM fig, SAM fig
      7. Ollama summary

    Returns 5 values for Gradio:
      heatmap_img, sam_img, result_badge, summary, sam_status
    """
    if image is None:
        empty = "⚠️  Please upload a retinal fundus image."
        yield None, None, empty, "", ""
        return

    if not MODELS:
        empty = "⚠️  No model weights found. Check .pth files."
        yield None, None, empty, "", ""
        return

    # ── 1. Classify ───────────────────────────────────────────────────────────
    label, pred_idx, confidence, img_tensor, all_probs = run_inference(image)
    orig_label = label

    # ── 2. Grad-CAM ───────────────────────────────────────────────────────────
    cam = ensemble_gradcam(img_tensor, pred_idx)

    # ── 3. Grad-CAM → SAM prompt ──────────────────────────────────────────────
    orig_w, orig_h = image.size
    box, centroid, gradcam_mask = get_sam_prompt_from_gradcam(cam, orig_w, orig_h)

    # ── 4. SAM segmentation ───────────────────────────────────────────────────
    img_np_orig = np.array(image.convert('RGB'))
    mask        = run_sam_segmentation(SAM_PREDICTOR, SAM_BACKEND,
                                       img_np_orig, box, centroid)
    sam_ok      = mask is not None

    # Fall back to Grad-CAM binary mask if SAM unavailable
    effective_mask = mask if sam_ok else (gradcam_mask if gradcam_mask is not None else None)

    # ── 5. Macular weighting ──────────────────────────────────────────────────
    central_frac, zone_label = compute_central_retina_weight(
        effective_mask, orig_h, orig_w
    )
    adj_probs, sam_note = adjust_probabilities_with_segmentation(
        all_probs, central_frac, zone_label, sam_ok
    )
    adj_idx   = max(adj_probs, key=adj_probs.get)
    adj_label = adj_idx
    adj_conf  = adj_probs[adj_idx]

    # ── 6a. Grad-CAM figure ───────────────────────────────────────────────────
    heatmap_img = make_heatmap_figure(image, cam, orig_label, confidence, all_probs)

    # ── 6b. SAM figure ────────────────────────────────────────────────────────
    sam_img = make_sam_figure(
        image, effective_mask, box,
        adj_probs, orig_label, adj_label,
        zone_label, sam_note
    )

    # ── 7. Badge ──────────────────────────────────────────────────────────────
    changed_str = ""
    if adj_label != orig_label:
        changed_str = (f"  ·  SAM adjusted → **{SEVERITY_META[adj_label]['label']}**"
                       f"  ({adj_conf:.1f}%)")

    sam_backend_str = SAM_BACKEND.upper() if SAM_BACKEND != 'none' else 'unavailable'
    badge_md = (
        f"## Initial:  {SEVERITY_META[orig_label]['label']}  ({confidence*100:.2f}%)"
        f"{changed_str}\n\n"
        f"**Severity level:** {SEVERITY_META[adj_label]['level']} / 4  |  "
        f"**Lesion zone:** {zone_label}  |  "
        f"**SAM backend:** {sam_backend_str}  |  "
        f"**Models:** {', '.join(MODELS.keys())}"
    )

    sam_status = (
        f"✅ SAM ({SAM_BACKEND}) mask generated — {zone_label} lesion, "
        f"macular overlap {central_frac*100:.0f}%"
        if sam_ok
        else "⚠️  SAM mask unavailable — using Grad-CAM region. Original classifier output shown."
    )

    # ── Yield before LLM to show images instantly ─────────────────────────────
    yield heatmap_img, sam_img, badge_md, "Generating clinical summary...", sam_status
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 8. Ollama ─────────────────────────────────────────────────────────────
    prompt  = build_prompt(orig_label, confidence, adj_probs, patient_history,
                           adj_label, zone_label, sam_note)
    
    summary = ""
    for chunk in query_ollama(prompt):
        summary += chunk
        yield heatmap_img, sam_img, badge_md, summary, sam_status


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

    gr.HTML("""
    <div style="text-align:center; padding:28px 0 12px;">
        <div style="font-size:12px; letter-spacing:3px; text-transform:uppercase;
                    color:#34d399; font-weight:600; margin-bottom:8px;">
            AI-Assisted Ophthalmology
        </div>
        <h1 style="font-size:2rem; font-weight:800; color:#f1f5f9; margin:0;">
            Diabetic Retinopathy Detection
        </h1>
        <p style="color:#64748b; margin-top:10px; font-size:0.9rem;">
            Ensemble classifier &nbsp;·&nbsp; Grad-CAM &nbsp;·&nbsp;
            SAM/SAM2 lesion segmentation &nbsp;·&nbsp; Macular weighting &nbsp;·&nbsp;
            LLaMA 3.2 clinical summary
        </p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── Left: inputs ──────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 🩺 Patient Input")

            image_input = gr.Image(
                type="pil", label="Retinal Fundus Image",
                image_mode="RGB", height=260,
                sources=["upload", "clipboard"],
            )
            history_input = gr.Textbox(
                label="Patient History",
                placeholder=(
                    "e.g. 58-year-old male, Type 2 diabetes 12 years, "
                    "HbA1c 9.2%, hypertensive, no prior laser treatment..."
                ),
                lines=6, max_lines=14,
            )
            analyse_btn = gr.Button("🔬  Analyse", variant="primary", size="lg")

            gr.HTML("""
            <div style="margin-top:12px; padding:12px; border-radius:8px;
                        background:#1e3a5f; border:1px solid #1d4ed8;
                        font-size:0.79rem; color:#93c5fd; line-height:1.7;">
                <strong>ℹ️ Setup checklist:</strong><br>
                • <code>.pth</code> weights in working directory<br>
                • SAM2 or SAM checkpoint in working directory<br>
                • <code>ollama serve</code> running + <code>llama3.2</code> pulled<br>
                <br>
                SAM is optional — falls back to Grad-CAM region if unavailable.
            </div>
            """)

        # ── Right: outputs ────────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=520):
            gr.Markdown("### 📊 Analysis Results")

            result_badge = gr.Markdown(
                value="_Upload an image and click Analyse._"
            )
            sam_status_md = gr.Markdown(value="")

            with gr.Tabs():
                with gr.Tab("🔥 Grad-CAM"):
                    heatmap_output = gr.Image(
                        label="Original · Grad-CAM overlay · Class probabilities",
                        type="pil", height=300, interactive=False,
                    )
                with gr.Tab("🔬 SAM Segmentation"):
                    sam_output = gr.Image(
                        label="SAM prompt · Lesion mask · Adjusted probabilities",
                        type="pil", height=300, interactive=False,
                    )

            gr.Markdown("### 🧠 Clinical Summary  *(LLaMA 3.2)*")
            summary_output = gr.Textbox(
                label="", lines=18, max_lines=40,
                interactive=False,
                placeholder="Clinical summary will appear here after analysis...",
            )

    gr.HTML("""
    <div style="text-align:center; padding:18px 0 6px;
                color:#475569; font-size:0.77rem;
                border-top:1px solid #1e293b; margin-top:18px;">
        ⚠️ Research &amp; clinical decision support only.
        Does not replace professional ophthalmological diagnosis.
    </div>
    """)

    analyse_btn.click(
        fn=analyse,
        inputs=[image_input, history_input],
        outputs=[heatmap_output, sam_output, result_badge, summary_output, sam_status_md],
        show_progress="full",
    )

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        theme=THEME,
        server_name="127.0.0.1",
        share=True,
        inbrowser=True,
    )