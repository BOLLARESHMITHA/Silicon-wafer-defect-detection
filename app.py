import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 9
CLASS_NAMES = [
    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
    'Loc', 'Near-full', 'Random', 'Scratch', 'none'
]

CNN_MODEL_PATH    = "best_cnn.pth"
VIT_MODEL_PATH    = "best_vit.pth"
HYBRID_MODEL_PATH = "best_hybrid.pth"


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────
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
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(embed_dim, embed_dim * 3)
        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = self.dropout((q @ k.transpose(-2,-1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1,2).reshape(B, N, C))


class ViTBlock(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = ViTAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class WaferViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3,
                 embed_dim=192, depth=12, num_heads=3,
                 mlp_ratio=4.0, num_classes=9, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        num_patches      = self.patch_embed.num_patches
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop    = nn.Dropout(dropout)
        self.blocks      = nn.Sequential(*[
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        B   = x.shape[0]
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = self.pos_drop(torch.cat([cls, x], dim=1) + self.pos_embed)
        x   = self.norm(self.blocks(x))
        return self.head(x[:, 0])


class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=9, img_size=64, d_model=256, nhead=8, num_layers=4, dropout=0.5):
        super().__init__()
        self.img_size = img_size
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128),  nn.ReLU(),
            nn.Conv2d(128, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
        )
        feat_h  = img_size // 4
        seq_len = feat_h * feat_h
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
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


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def get_model_input_size(model):
    if hasattr(model, 'patch_embed'):
        num_patches = model.patch_embed.num_patches
        patch_size  = model.patch_embed.proj.kernel_size[0]
        return int(num_patches ** 0.5) * patch_size
    if hasattr(model, 'img_size'):
        return model.img_size
    return 64


def get_yolo_bbox_largest_defect(wafer):
    arr = np.array(wafer, dtype=np.uint8)
    defect_mask = np.where(arr == 2, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    img_h, img_w = arr.shape
    return [(x + w / 2.0) / img_w, (y + h / 2.0) / img_h, w / img_w, h / img_h]


@st.cache_resource
def load_models():
    cnn    = WaferCNN(NUM_CLASSES).to(DEVICE)
    vit    = WaferViT(num_classes=NUM_CLASSES).to(DEVICE)
    hybrid = HybridCNNTransformer(NUM_CLASSES, dropout=0.5).to(DEVICE)

    cnn.load_state_dict(torch.load(CNN_MODEL_PATH,    map_location=DEVICE, weights_only=True))
    vit.load_state_dict(torch.load(VIT_MODEL_PATH,    map_location=DEVICE, weights_only=True))
    hybrid.load_state_dict(torch.load(HYBRID_MODEL_PATH, map_location=DEVICE, weights_only=True))

    cnn.eval(); vit.eval(); hybrid.eval()
    return {"CNN": cnn, "ViT": vit, "Hybrid": hybrid}


def preprocess_image(img_array, target_size):
    """Normalise → resize → return float32 RGB ndarray and tensor."""
    img = img_array.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img_resized = cv2.resize(
        (img * 255).astype(np.uint8),
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST
    )
    img_f  = img_resized.astype(np.float32) / 255.0
    tensor = torch.tensor(img_f).permute(2, 0, 1).unsqueeze(0).to(DEVICE).float()
    return img_resized, tensor


def build_raw_wafer(img_array):
    """Recover 0/1/2 wafer encoding from a rendered PNG."""
    if img_array.max() <= 1.0:
        img_uint8 = (img_array * 255).astype(np.uint8)
    else:
        img_uint8 = img_array.astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    raw  = np.zeros_like(gray, dtype=np.uint8)
    raw[gray > 30]  = 1
    raw[gray > 180] = 2
    return raw


def predict(model, img_array):
    target_size          = get_model_input_size(model)
    img_resized, tensor  = preprocess_image(img_array, target_size)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    pred_cls   = CLASS_NAMES[np.argmax(probs)]
    confidence = float(probs.max())
    return pred_cls, confidence, probs, img_resized


def draw_result_figure(img_rgb, probs, pred_cls, confidence, model_name, raw_wafer):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#0e1117')
    for ax in axes:
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # Panel 1 — input image
    axes[0].imshow(img_rgb, cmap='plasma')
    axes[0].set_title(f"{model_name} input ({img_rgb.shape[0]}×{img_rgb.shape[1]})",
                      color='white', fontsize=10)
    axes[0].axis('off')

    # Panel 2 — bounding box
    axes[1].imshow(img_rgb, cmap='plasma')
    axes[1].set_title("Defect bounding box", color='white', fontsize=10)
    axes[1].axis('off')

    if raw_wafer is not None:
        bbox = get_yolo_bbox_largest_defect(raw_wafer)
        if bbox is not None:
            h, w = img_rgb.shape[:2]
            xc, yc, bw, bh = bbox
            rect = patches.Rectangle(
                ((xc - bw / 2) * w, (yc - bh / 2) * h), bw * w, bh * h,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[1].add_patch(rect)

    # Panel 3 — probability bar chart
    colors = ['#e74c3c' if c == pred_cls else '#3498db' for c in CLASS_NAMES]
    axes[2].barh(CLASS_NAMES, probs, color=colors)
    axes[2].set_xlim(0, 1)
    axes[2].set_title(f"Prediction: {pred_cls}  ({confidence:.1%})",
                      color='white', fontsize=10)
    axes[2].tick_params(colors='white')
    axes[2].xaxis.label.set_color('white')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Defect Classifier",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Wafer Defect Classifier")
st.markdown(
    "Upload a wafer map image and classify its defect type using **CNN**, **ViT**, or the **Hybrid** model."
)

# Sidebar — model selection
st.sidebar.header("⚙️ Settings")
selected_models = st.sidebar.multiselect(
    "Models to run",
    options=["CNN", "ViT", "Hybrid"],
    default=["CNN", "ViT", "Hybrid"]
)

# Load models
with st.spinner("Loading models…"):
    try:
        models = load_models()
        st.sidebar.success("✅ All models loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Model load failed: {e}")
        st.stop()

# File upload
uploaded_file = st.file_uploader(
    "Upload a wafer map image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Decode uploaded image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    bgr        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    raw_wafer = build_raw_wafer(img_rgb)

    st.subheader("Uploaded wafer map")
    st.image(img_rgb, width=250, clamp=True)

    if not selected_models:
        st.warning("Please select at least one model in the sidebar.")
    else:
        st.subheader("Predictions")
        for model_name in selected_models:
            model = models[model_name]
            with st.spinner(f"Running {model_name}…"):
                pred_cls, confidence, probs, img_resized = predict(model, img_rgb)

            fig = draw_result_figure(
                img_resized, probs, pred_cls, confidence, model_name, raw_wafer
            )

            with st.expander(
                f"**{model_name}** → {pred_cls}  ({confidence:.1%})",
                expanded=True
            ):
                st.pyplot(fig)
                plt.close(fig)

        # Summary comparison table
        if len(selected_models) > 1:
            st.subheader("📊 Model Comparison")
            rows = []
            for model_name in selected_models:
                pc, conf, _, _ = predict(models[model_name], img_rgb)
                rows.append({"Model": model_name, "Prediction": pc, "Confidence": f"{conf:.1%}"})
            import pandas as pd
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
else:
    st.info("👆 Upload a wafer map image to get started.")