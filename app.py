"""
Silicon Wafer Defect Detection — Streamlit App
Light theme with multi-model comparison and improved UI
"""

import io
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
               'Loc', 'Near-full', 'Random', 'Scratch', 'none']

DEFECT_INFO = {
    "Center":     {"emoji": "🎯", "desc": "Defects clustered around the wafer's center, often linked to centrifugal process steps (e.g. spin-coating, CMP)."},
    "Donut":      {"emoji": "🍩", "desc": "A ring-shaped defect band offset from the very center — frequently tied to uneven etch or deposition profiles."},
    "Edge-Loc":   {"emoji": "📍", "desc": "Localized defect clusters near the wafer edge, often from edge-bead or handling-related contamination."},
    "Edge-Ring":  {"emoji": "⭕", "desc": "A defect ring concentrated at the wafer's outer edge — classically associated with edge exclusion zone issues."},
    "Loc":        {"emoji": "🔘", "desc": "A localized defect cluster somewhere on the wafer surface, not tied to center or edge geometry."},
    "Near-full":  {"emoji": "⚠️", "desc": "Defects covering nearly the entire wafer — usually indicates a severe, global process failure."},
    "Random":     {"emoji": "🎲", "desc": "Defects scattered with no clear spatial pattern — often particle contamination."},
    "Scratch":    {"emoji": "📏", "desc": "A linear defect trail, typically caused by mechanical handling or wafer-to-wafer contact."},
    "none":       {"emoji": "✅", "desc": "No significant defect pattern detected — wafer appears clean."},
}

# ──────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS (must match training architectures exactly)
# ──────────────────────────────────────────────────────────────────────────────
class WaferCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTAttention(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.dropout((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class ViTBlock(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ViTAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class WaferViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=192,
                 depth=12, num_heads=3, mlp_ratio=4.0, num_classes=9, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[ViTBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = self.pos_drop(torch.cat([cls, x], dim=1) + self.pos_embed)
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])


class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=9, img_size=64, d_model=256, nhead=8, num_layers=4, dropout=0.5):
        super().__init__()
        self.img_size = img_size
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
        )
        feat_h = img_size // 4
        seq_len = feat_h * feat_h
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                     dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.cnn_backbone(x)
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)
        feat = feat + self.pos_embed[:, :feat.size(1), :]
        feat = self.transformer(feat)
        feat = feat.mean(dim=1)
        return self.head(feat)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    try:
        cnn = WaferCNN(9).to(DEVICE)
        cnn.load_state_dict(torch.load("best_cnn.pth", map_location=DEVICE))
        cnn.eval()
        models["CNN"] = cnn
    except Exception as e:
        st.warning(f"Could not load CNN weights: {e}")

    try:
        vit = WaferViT(num_classes=9).to(DEVICE)
        vit.load_state_dict(torch.load("best_vit.pth", map_location=DEVICE))
        vit.eval()
        models["ViT"] = vit
    except Exception as e:
        st.warning(f"Could not load ViT weights: {e}")

    try:
        hybrid = HybridCNNTransformer(9).to(DEVICE)
        hybrid.load_state_dict(torch.load("best_hybrid.pth", map_location=DEVICE))
        hybrid.eval()
        models["Hybrid"] = hybrid
    except Exception as e:
        st.warning(f"Could not load Hybrid weights: {e}")

    return models


def get_model_input_size(model):
    if hasattr(model, "patch_embed"):
        num_patches = model.patch_embed.num_patches
        patch_size = model.patch_embed.proj.kernel_size[0]
        return int(num_patches ** 0.5) * patch_size
    if hasattr(model, "img_size"):
        return model.img_size
    return 64


def get_largest_defect_bbox(raw_wafer):
    arr = np.array(raw_wafer, dtype=np.uint8)
    defect_mask = np.where(arr == 2, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    img_h, img_w = arr.shape
    return [(x + w / 2.0) / img_w, (y + h / 2.0) / img_h, w / img_w, h / img_h]


def predict(model, model_name, pil_img):
    img = np.array(pil_img.convert("RGB")).astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    target_size = get_model_input_size(model)
    img_resized = cv2.resize((img * 255).astype(np.uint8), (target_size, target_size),
                              interpolation=cv2.INTER_NEAREST)
    img_f = img_resized.astype(np.float32) / 255.0

    gray_raw = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    raw_wafer = np.zeros_like(gray_raw, dtype=np.uint8)
    raw_wafer[gray_raw > 30] = 1
    raw_wafer[gray_raw > 180] = 2

    tensor = torch.tensor(img_f).permute(2, 0, 1).unsqueeze(0).to(DEVICE).float()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_cls = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    bbox = get_largest_defect_bbox(raw_wafer)

    return {
        "model_name": model_name,
        "img_resized": img_resized,
        "target_size": target_size,
        "pred_cls": pred_cls,
        "confidence": confidence,
        "probs": probs,
        "bbox": bbox,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LIGHT THEME STYLES
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .hero-title { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #f8f9fa 0%, #f0f2f5 100%);
    color: #1a1a1a;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f5f7fa 100%);
    border-right: 1px solid #e0e4e8;
}

/* HERO */
.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 20px;
    background: linear-gradient(120deg, rgba(99, 102, 241, 0.1), rgba(59, 130, 246, 0.1));
    border: 1px solid rgba(99, 102, 241, 0.3);
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08);
    margin-bottom: 1.6rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #1f2937;
    margin: 0;
}
.hero-sub {
    font-size: 1.1rem;
    color: #4b5563;
    margin-top: 0.8rem;
    line-height: 1.6;
}

/* GLASS CARD */
.glass-card {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1.8rem;
    box-shadow: 0 2px 15px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
}

.section-label {
    font-size: 0.9rem;
    font-weight: 700;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 1rem;
}

/* BADGES */
.defect-badge {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(59, 130, 246, 0.2));
    border: 1.5px solid #6366f1;
    color: #1f2937;
    padding: 0.5rem 1.2rem;
    border-radius: 24px;
    font-weight: 600;
    display: inline-block;
    font-size: 0.95rem;
}

.conf-pill {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.15));
    border: 1.5px solid #22c55e;
    color: #15803d;
    padding: 0.5rem 1.2rem;
    border-radius: 24px;
    font-weight: 600;
    display: inline-block;
    margin-left: 0.8rem;
    font-size: 0.95rem;
}

.model-pill {
    background: linear-gradient(135deg, rgba(249, 115, 22, 0.15), rgba(234, 179, 8, 0.15));
    border: 1px solid #f97316;
    color: #92400e;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
}

/* QA BLOCK */
.qa-block {
    background: rgba(249, 250, 251, 0.7);
    border-left: 4px solid #6366f1;
    padding: 1.2rem;
    border-radius: 8px;
    margin-top: 1rem;
}
.qa-question {
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 0.8rem;
}
.qa-answer {
    color: #4b5563;
    line-height: 1.6;
}

/* STREAMLIT OVERRIDES */
[data-testid="stCheckbox"] { color: #1a1a1a; }
[data-testid="stSelectbox"] { color: #1a1a1a; }
[data-testid="stMultiSelect"] { color: #1a1a1a; }

.stButton>button {
    background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
}

/* MODEL COMPARISON GRID */
.model-comparison {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
}

.model-card {
    background: white;
    border: 1.5px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
    transition: all 0.3s ease;
}

.model-card:hover {
    border-color: #6366f1;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
    transform: translateY(-2px);
}

.model-name {
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.8rem;
    font-size: 1.1rem;
}

.model-prediction {
    margin: 0.8rem 0;
}

.model-confidence {
    font-size: 0.9rem;
    color: #6b7280;
    margin-top: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Wafer Inspector")
    st.markdown("Configure your analysis below.")
    st.markdown("---")

    models = load_models()
    if not models:
        st.error("No model weights found. Make sure best_cnn.pth, best_vit.pth, and best_hybrid.pth are in the app directory.")
        st.stop()

    # Multi-model selection
    selected_models = st.multiselect(
        "Select models to compare",
        options=list(models.keys()),
        default=list(models.keys()),
        help="Choose one or more models to run predictions"
    )
    
    if not selected_models:
        st.warning("Please select at least one model!")
        st.stop()

    st.markdown(f"<span class='model-pill'>Running on {DEVICE.upper()}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 📋 Defect Classes")
    with st.expander("View all defect types", expanded=False):
        for name, info in DEFECT_INFO.items():
            st.markdown(f"**{info['emoji']} {name}**")
            st.caption(info['desc'])

# ──────────────────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Silicon Wafer Defect Detection</div>
    <div class="hero-sub">
        Upload a wafer map and let AI models classify the defect pattern across 9 categories — 
        compare multiple architectures (CNN, ViT, Hybrid) simultaneously for comprehensive analysis.
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN — UPLOAD
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📤 Input</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a wafer map image", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")
show_bbox = st.checkbox("Show defect bounding-box overlay", value=True)
st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# RESULTS SECTION (only shows when image is uploaded)
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    pil_img = Image.open(io.BytesIO(uploaded_file.read()))
    
    # Show original image
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📸 Uploaded Image</div>', unsafe_allow_html=True)
    st.image(pil_img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Run predictions for selected models
    with st.spinner("🔄 Running inference on selected models..."):
        results = {}
        for model_name in selected_models:
            model = models[model_name]
            results[model_name] = predict(model, model_name, pil_img)

    # Display results for each model in a grid
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🎯 Predictions</div>', unsafe_allow_html=True)
    
    cols = st.columns(len(selected_models))
    for col, model_name in enumerate(selected_models):
        result = results[model_name]
        info = DEFECT_INFO.get(result["pred_cls"], {"emoji": "❓", "desc": ""})
        
        with cols[col]:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div class="model-name">{model_name}</div>
                <div class="model-prediction">
                    <span class="defect-badge" style="font-size:0.9rem;">{info['emoji']} {result['pred_cls']}</span>
                </div>
                <div class="model-confidence">{result['confidence']*100:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Detailed analysis for first selected model
    primary_model = selected_models[0]
    primary_result = results[primary_model]
    primary_info = DEFECT_INFO.get(primary_result["pred_cls"], {"emoji": "❓", "desc": ""})

    # Image with bounding box
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🔍 Detailed Analysis</div>', unsafe_have_html=True)
    
    disp_img = primary_result["img_resized"].copy()
    if show_bbox and primary_result["bbox"] is not None:
        h, w = disp_img.shape[:2]
        xc, yc, bw, bh = primary_result["bbox"]
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)
        cv2.rectangle(disp_img, (x1, y1), (x2, y2), (0, 150, 255), 3)
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.image(disp_img, caption=f"{primary_model} Input ({primary_result['target_size']}×{primary_result['target_size']})",
                 use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div style="padding: 1.5rem 0;">
            <h3 style="color: #1f2937; margin-top: 0;">Primary Defect</h3>
            <div style="margin: 1rem 0;">
                <span class="defect-badge">{primary_info['emoji']} {primary_result['pred_cls']}</span>
                <span class="conf-pill">{primary_result['confidence']*100:.1f}%</span>
            </div>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(99, 102, 241, 0.05); border-radius: 10px; border-left: 3px solid #6366f1;">
                <p style="margin: 0; color: #4b5563; line-height: 1.6;">{primary_info['desc']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Confidence distribution chart
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 Class Confidence Distribution</div>', unsafe_have_html=True)
    
    sorted_idx = np.argsort(primary_result["probs"])
    fig = go.Figure(go.Bar(
        x=primary_result["probs"][sorted_idx],
        y=[CLASS_NAMES[i] for i in sorted_idx],
        orientation="h",
        marker=dict(
            color=primary_result["probs"][sorted_idx],
            colorscale=[[0, "#e5e7eb"], [1, "#6366f1"]],
        ),
        text=[f"{p*100:.1f}%" for p in primary_result["probs"][sorted_idx]],
        textposition="outside",
    ))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#4b5563", size=12),
        xaxis=dict(range=[0, 1], showgrid=True, gridcolor="#e5e7eb", tickformat=".0%"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # All models comparison table
    if len(selected_models) > 1:
        st.markdown('<div class="glass-card">', unsafe_have_html=True)
        st.markdown('<div class="section-label">⚖️ Model Comparison</div>', unsafe_have_html=True)
        
        comparison_data = []
        for model_name in selected_models:
            r = results[model_name]
            comparison_data.append({
                "Model": model_name,
                "Prediction": r["pred_cls"],
                "Confidence": f"{r['confidence']*100:.1f}%",
                "Input Size": f"{r['target_size']}×{r['target_size']}"
            })
        
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_have_html=True)

# Footer
st.markdown(
    "<div style='text-align:center; color:#9ca3af; padding:2rem 0; font-size:0.85rem;'>"
    "🔬 Silicon Wafer Defect Detection · Multi-Model AI Analysis<br>"
    "CNN | Vision Transformer | Hybrid CNN-Transformer"
    "</div>",
    unsafe_allow_html=True,
)
