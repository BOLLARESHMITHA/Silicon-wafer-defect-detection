"""
Silicon Wafer Defect Detection — Streamlit App
Polished UI: gradient hero, glass cards, defect-tag badges, bbox overlay, confidence chart.
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
        models["Hybrid (CNN+Transformer)"] = hybrid
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
# STYLES
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .hero-title { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: radial-gradient(circle at 15% 0%, #1b1033 0%, #0d0a1f 45%, #07050f 100%);
    color: #e8e6f4;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #150f2b 0%, #0c0918 100%);
    border-right: 1px solid rgba(150,120,255,0.15);
}

/* HERO */
.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 22px;
    background: linear-gradient(120deg, rgba(124,58,237,0.35), rgba(34,211,238,0.15));
    border: 1px solid rgba(168,140,255,0.25);
    box-shadow: 0 8px 40px rgba(99,60,255,0.18);
    margin-bottom: 1.6rem;
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #c7b8ff, #8ef6ff);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: #b7b0d4;
    font-size: 1.02rem;
    max-width: 760px;
}

/* CARD */
.glass-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(168,140,255,0.18);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(6px);
    margin-bottom: 1.2rem;
}

/* BADGE */
.defect-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.55rem 1.1rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 1.05rem;
    background: linear-gradient(90deg, rgba(34,211,238,0.18), rgba(124,58,237,0.18));
    border: 1px solid rgba(94, 234, 212, 0.45);
    color: #9dffe8;
}

.conf-pill {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(168,140,255,0.18);
    border: 1px solid rgba(168,140,255,0.4);
    color: #d6c9ff;
    margin-left: 0.5rem;
}

.section-label {
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8d84b8;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.qa-block {
    border-left: 3px solid #7c3aed;
    padding-left: 1rem;
    margin-top: 0.8rem;
}
.qa-question { color: #c7b8ff; font-weight: 600; margin-bottom: 0.3rem; }
.qa-answer { color: #d8d4ec; line-height: 1.55; }

footer, header[data-testid="stHeader"] { background: transparent; }

div[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.03);
    border: 1.5px dashed rgba(168,140,255,0.4);
    border-radius: 16px;
}

.model-pill {
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    background: rgba(34,211,238,0.12);
    border: 1px solid rgba(34,211,238,0.35);
    color: #aef0ff;
    font-size: 0.82rem;
    font-weight: 600;
    display: inline-block;
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

    model_choice = st.selectbox("Model architecture", list(models.keys()), index=0)
    st.markdown(f"<span class='model-pill'>Running on {DEVICE.upper()}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Defect classes")
    for name, info in DEFECT_INFO.items():
        st.markdown(f"{info['emoji']} **{name}**")

# ──────────────────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Silicon Wafer Defect Detection</div>
    <div class="hero-sub">
        Upload a wafer map and let a CNN, Vision Transformer, or Hybrid CNN+Transformer
        model classify the defect pattern across 9 categories — with bounding-box localization
        and full confidence breakdown.
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN — UPLOAD
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.3], gap="large")

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a wafer map image", type=["png", "jpg", "jpeg", "bmp"])

    sample_note = st.checkbox("Use defect bounding-box overlay", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        pil_img = Image.open(io.BytesIO(uploaded_file.read()))
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Original Upload</div>', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    if uploaded_file is None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
        st.info("Upload a wafer map on the left to run a defect classification.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Running inference..."):
            model = models[model_choice]
            result = predict(model, model_choice, pil_img)

        info = DEFECT_INFO.get(result["pred_cls"], {"emoji": "❓", "desc": ""})

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            f"""<span class="defect-badge">{info['emoji']} {result['pred_cls']}</span>
            <span class="conf-pill">{result['confidence']*100:.1f}% confidence</span>""",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div style='margin-top:0.8rem; color:#bcb6d8;'>{info['desc']}</div>", unsafe_allow_html=True)

        # bbox overlay image
        disp_img = result["img_resized"].copy()
        if sample_note and result["bbox"] is not None:
            h, w = disp_img.shape[:2]
            xc, yc, bw, bh = result["bbox"]
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            disp_img = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)
            cv2.rectangle(disp_img, (x1, y1), (x2, y2), (0, 60, 255), 2)
            disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.image(disp_img, caption=f"{result['model_name']} input — {result['target_size']}×{result['target_size']}",
                  use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Confidence chart
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Class Confidence</div>', unsafe_allow_html=True)
        sorted_idx = np.argsort(result["probs"])
        fig = go.Figure(go.Bar(
            x=result["probs"][sorted_idx],
            y=[CLASS_NAMES[i] for i in sorted_idx],
            orientation="h",
            marker=dict(
                color=result["probs"][sorted_idx],
                colorscale=[[0, "#3a2a6b"], [1, "#22d3ee"]],
            ),
            text=[f"{p*100:.1f}%" for p in result["probs"][sorted_idx]],
            textposition="outside",
        ))
        fig.update_layout(
            height=340,
            margin=dict(l=10, r=30, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d8d4ec"),
            xaxis=dict(range=[0, 1], showgrid=False, tickformat=".0%"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Q&A explanation block (matches the reference screenshot style)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Explanation</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="qa-block">
            <div class="qa-question">Question: What is the primary defect pattern in this wafer map?</div>
            <div class="qa-answer">
                Answer: The primary defect observed in this wafer is a
                <span class="defect-badge" style="font-size:0.95rem; padding:0.3rem 0.7rem;">
                    {info['emoji']} {result['pred_cls']}
                </span> defect, predicted with {result['confidence']*100:.1f}% confidence
                by the {result['model_name']} model.
                {info['desc']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON (optional, runs all loaded models)
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None and len(models) > 1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Compare All Models</div>', unsafe_allow_html=True)
    cols = st.columns(len(models))
    for col, (name, m) in zip(cols, models.items()):
        with col:
            r = predict(m, name, pil_img)
            i = DEFECT_INFO.get(r["pred_cls"], {"emoji": "❓"})
            st.markdown(f"**{name}**")
            st.markdown(
                f"<span class='defect-badge' style='font-size:0.9rem;'>{i['emoji']} {r['pred_cls']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"{r['confidence']*100:.1f}% confidence")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center; color:#6e6790; padding:1.5rem 0; font-size:0.85rem;'>"
    "Silicon Wafer Defect Detection · CNN / ViT / Hybrid CNN-Transformer</div>",
    unsafe_allow_html=True,
)
