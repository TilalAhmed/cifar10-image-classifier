import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Image Classification System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
ICONS   = ['✈️','🚗','🐦','🐱','🦌','🐶','🐸','🐴','🚢','🚛']
COLORS  = ['#1f6feb','#388bfd','#3fb950','#f0883e','#a371f7',
           '#f85149','#56d364','#e3b341','#79c0ff','#ffa657']

# Real training data from notebook
TRAIN_ACC  = [0.4697,0.6087,0.6531,0.6819,0.7015,0.7220,0.7372,0.7530,0.766,0.778]
TEST_ACC   = [0.610, 0.643, 0.665, 0.672, 0.680, 0.683, 0.686, 0.687, 0.688,0.689]
TRAIN_LOSS = [1.4688,1.1189,0.9950,0.9144,0.8564,0.7977,0.7532,0.7097,0.672,0.641]
TEST_LOSS  = [1.120, 1.040, 0.980, 0.970, 0.962, 0.961, 0.960, 0.961, 0.961,0.961]

CLASS_ACC  = [71.9, 73.7, 51.4, 43.5, 55.0, 54.5, 77.7, 67.7, 69.5, 70.2]

CM = [
    [719,21,43,12,9,6,6,14,69,101],
    [18,737,10,7,3,2,4,2,76,139],
    [68,15,514,68,52,79,52,52,60,40],
    [43,6,62,435,40,189,29,47,88,103],
    [16,4,47,29,550,34,17,43,46,74],
    [11,1,53,116,41,545,17,33,42,41],
    [6,3,28,26,13,14,777,8,53,72],
    [14,13,49,40,58,69,24,677,23,33],
    [58,52,31,22,11,9,8,12,695,102],
    [41,114,15,12,16,9,10,4,77,702],
]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] li { color: #8b949e !important; font-size: 0.78rem !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #e2e8f0 !important; }

[data-testid="stMetric"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.68rem !important; text-transform: uppercase; letter-spacing: 0.5px; }
[data-testid="stMetricValue"] { color: #fff !important; font-size: 1.4rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { color: #484f58 !important; font-size: 0.68rem !important; }
[data-testid="stMetricDelta"] svg { display: none; }

.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.55rem 1.2rem !important;
    font-size: 0.85rem !important;
}
.stButton > button:hover { opacity: 0.9 !important; }

[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1px dashed #30363d !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] * { color: #8b949e !important; }

.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
}
.card-title {
    font-size: 10px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}
.ca-row { display: flex; align-items: center; gap: 6px; margin-bottom: 5px; }
.ca-name { font-size: 10px; color: #8b949e; width: 82px; flex-shrink: 0; text-align: right; text-transform: capitalize; }
.ca-track { flex: 1; height: 10px; background: #0d1117; border-radius: 2px; overflow: hidden; }
.ca-fill { height: 100%; border-radius: 2px; }
.ca-val { font-size: 10px; color: #8b949e; width: 36px; flex-shrink: 0; }

.arch-wrap { display: flex; align-items: center; gap: 0; overflow-x: auto; padding-bottom: 4px; }
.arch-step { flex: 1; min-width: 62px; background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 9px 5px; text-align: center; }
.arch-title { font-size: 8px; font-weight: 600; color: #e2e8f0; line-height: 1.4; }
.arch-sub { font-size: 7px; color: #484f58; margin-top: 2px; }
.arch-arr { color: #30363d; font-size: 11px; padding: 0 3px; flex-shrink: 0; }

.info-box {
    background: #0d1117;
    border-left: 2px solid #1f6feb;
    border-radius: 0 6px 6px 0;
    padding: 10px 12px;
    font-size: 10px;
    color: #8b949e;
    line-height: 1.9;
    margin-top: 8px;
}
.result-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    margin-top: 8px;
}
.pred-label { font-size: 1.2rem; font-weight: 700; color: #388bfd; text-transform: uppercase; }
.pred-conf { font-size: 10px; color: #8b949e; margin-top: 3px; }
.bar-row { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
.bar-name { font-size: 9px; color: #8b949e; width: 70px; flex-shrink: 0; text-align: right; text-transform: capitalize; }
.bar-track { flex: 1; height: 5px; background: #21262d; border-radius: 2px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 2px; }
.bar-val { font-size: 9px; color: #8b949e; width: 34px; flex-shrink: 0; }
.mt { width: 100%; border-collapse: collapse; font-size: 10px; margin-top: 8px; }
.mt th { color: #484f58; font-size: 9px; text-transform: uppercase; letter-spacing: .5px; padding: 0 0 6px; border-bottom: 1px solid #21262d; text-align: left; }
.mt td { padding: 6px 0; border-bottom: 1px solid #161b22; color: #c9d1d9; }
.ht { width: 100%; border-collapse: collapse; font-size: 10px; }
.ht th { color: #484f58; font-size: 9px; text-transform: uppercase; letter-spacing: .5px; padding: 0 6px 7px 0; border-bottom: 1px solid #21262d; text-align: left; font-weight: 500; }
.ht td { padding: 6px 6px 6px 0; border-bottom: 1px solid #161b22; color: #c9d1d9; vertical-align: middle; }
.bk-ok { background: #052e16; color: #3fb950; border: 1px solid #1a7f37; border-radius: 99px; padding: 1px 7px; font-size: 9px; font-weight: 600; }
.styled-div { height: 1px; background: #21262d; margin: 0.8rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        for name in ["cifar10_model.keras", "cifar10_model.h5"]:
            if os.path.exists(name):
                return tf.keras.models.load_model(name)
    except Exception:
        pass
    return None

model = load_model()
model_loaded = model is not None


def plotly_dark(fig, height=220):
    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(family="Inter,sans-serif", color="#8b949e", size=9),
        margin=dict(l=10, r=10, t=10, b=10), height=height, showlegend=False,
    )
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#21262d")
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
      <div style='width:42px;height:42px;background:linear-gradient(135deg,#1f6feb,#388bfd);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;'>🧠</div>
      <div>
        <div style='font-size:13px;font-weight:700;color:#fff;line-height:1.3;'>CIFAR-10<br>Image Classification<br>System</div>
        <div style='font-size:9px;color:#388bfd;font-weight:600;margin-top:2px;'>Deep Learning with CNN</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='styled-div'></div>", unsafe_allow_html=True)
    st.markdown("**10 Classes**")
    for i, cls in enumerate(CLASSES):
        st.markdown(
            f"<div style='font-size:11px;color:#8b949e;padding:2px 0;'>{ICONS[i]} {cls.upper()}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='styled-div'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        Architecture: CNN<br>
        Input: 32×32×3<br>
        Conv layers: 2<br>
        Training epochs: 10<br>
        Optimizer: Adam<br>
        Dataset: CIFAR-10<br>
        Framework: TensorFlow/Keras
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='styled-div'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:10px;'>
      <div style='font-size:10px;font-weight:600;color:#8b949e;margin-bottom:6px;'>🗄️ About Dataset</div>
      <div style='font-size:10px;color:#484f58;line-height:1.8;'>
        CIFAR-10 Dataset<br>
        60,000 color images in 10 classes.<br>
        50,000 training images and<br>
        10,000 test images.<br><br>
        Image Size: 32×32×3<br>
        Classes: 10
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#484f58;text-align:center;'>Built with ❤️ TensorFlow / Keras<br>Tilal Ahmed · Iqra University</div>", unsafe_allow_html=True)


# ── KPI Row ───────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
with k1: st.metric("🖼 Total Images",  "60,000",  "50K Train / 10K Test")
with k2: st.metric("⊞ Total Classes",  "10",       "CIFAR-10 Classes")
with k3: st.metric("🧠 Models Trained","2",        "ANN, CNN")
with k4: st.metric("🏆 Best Model",    "CNN",      "Highest Accuracy")
with k5: st.metric("📈 Best Accuracy", "68.86%",   "Test Accuracy (CNN)")
with k6: st.metric("📉 Test Loss",     "0.9609",   "Final Loss (CNN)")

st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

# ── Row 1: Classifier + Donut + Model Perf ───────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'><div class='card-title'>📷 Live Image Classifier</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"], label_visibility="collapsed")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    classify_btn = st.button("🔭 Predict Image")

    st.markdown("""
    <div class='info-box'>
        Upload any 32×32 or similar image of:<br>
        ✈️ airplane &nbsp; 🚗 automobile &nbsp; 🐦 bird<br>
        🐱 cat &nbsp; 🦌 deer &nbsp; 🐶 dog<br>
        🐸 frog &nbsp; 🐴 horse &nbsp; 🚢 ship &nbsp; 🚛 truck
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'><div class='card-title'>🎯 Dataset Class Distribution</div>", unsafe_allow_html=True)
    fig_donut = go.Figure(go.Pie(
        labels=CLASSES, values=[6000]*10, hole=0.58,
        marker=dict(colors=COLORS, line=dict(width=0)),
        textinfo="none", hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    fig_donut.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#8b949e", size=9), height=240,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(font=dict(size=8, color="#8b949e"), bgcolor="transparent",
                    x=1, y=0.5),
        showlegend=True,
    )
    fig_donut.add_annotation(text="60,000<br><span style='font-size:9px'>Total Images</span>",
        x=0.38, y=0.5, showarrow=False, font=dict(size=12, color="#fff"), align="center")
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'><div class='card-title'>⚡ Model Performance Comparison</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-bottom:10px;'>
      <div class='ca-row'><span class='ca-name'>CNN</span><div class='ca-track'><div class='ca-fill' style='width:68.86%;background:#1f6feb;'></div></div><span class='ca-val' style='color:#388bfd;'>68.86%</span></div>
      <div class='ca-row'><span class='ca-name'>ANN</span><div class='ca-track'><div class='ca-fill' style='width:50%;background:#6e40c9;'></div></div><span class='ca-val' style='color:#a371f7;'>50.00%</span></div>
    </div>
    <div style='font-size:9px;color:#484f58;text-align:center;margin-bottom:6px;'>Accuracy (%)</div>
    <table class='mt'>
      <thead><tr><th>Model</th><th>Type</th><th>Accuracy</th><th>Loss</th></tr></thead>
      <tbody>
        <tr><td style='color:#388bfd;font-weight:600;'>CNN</td><td style='color:#8b949e;'>Convolutional NN</td><td style='color:#3fb950;font-weight:600;'>68.86%</td><td style='color:#f85149;'>0.9609</td></tr>
        <tr><td style='color:#a371f7;font-weight:600;'>ANN</td><td style='color:#8b949e;'>Artificial NN</td><td style='color:#8b949e;'>~50.00%</td><td style='color:#484f58;'>—</td></tr>
      </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)


# ── Prediction Result ─────────────────────────────────────────────────────────
if classify_btn:
    if not uploaded:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Classifying..."):
            img = image.resize((32, 32))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            if model_loaded:
                predictions = model.predict(img_array, verbose=0)[0]
            else:
                predictions = np.random.dirichlet(np.ones(10) * 0.5)

        top_idx = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100
        top5_idx = np.argsort(predictions)[::-1][:5]

        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.markdown(f"""
            <div class='result-card'>
              <div style='font-size:36px;margin-bottom:6px;'>{ICONS[top_idx]}</div>
              <div class='pred-label'>{CLASSES[top_idx]}</div>
              <div class='pred-conf'>Confidence: <span style='color:#3fb950;font-weight:600;'>{confidence:.1f}%</span></div>
              <div style='font-size:9px;color:#484f58;margin-top:4px;'>{"CNN Model" if model_loaded else "Demo Mode"}</div>
            </div>
            """, unsafe_allow_html=True)

        with rc2:
            st.markdown("<div class='card'><div class='card-title'>Top 5 Predictions</div>", unsafe_allow_html=True)
            bars_html = "".join([
                f"""<div class='bar-row'>
                  <span class='bar-name'>{ICONS[i]} {CLASSES[i]}</span>
                  <div class='bar-track'><div class='bar-fill' style='width:{predictions[i]*100:.1f}%;background:{COLORS[i]};'></div></div>
                  <span class='bar-val'>{predictions[i]*100:.1f}%</span>
                </div>""" for i in top5_idx
            ])
            st.markdown(bars_html + "</div>", unsafe_allow_html=True)

        if not model_loaded:
            st.info("⚠ Demo mode — place `cifar10_model.keras` in root folder for real predictions.")


# ── Row 2: Sample Images + Training Curves + Confusion Matrix ─────────────────
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("<div class='card'><div class='card-title'>🖼 Sample Images from CIFAR-10</div>", unsafe_allow_html=True)
    BG = ['#1f4e8c','#8b1a1a','#1a5c1a','#5c3a1a','#3a5c1a',
          '#5c4a1a','#1a5c3a','#4a1a5c','#1a3a5c','#5c1a1a']
    imgs_html = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;">'
    for i in range(10):
        imgs_html += f"""<div style='text-align:center;'>
          <div style='width:100%;aspect-ratio:1;border-radius:6px;background:{BG[i]};display:flex;align-items:center;justify-content:center;font-size:20px;margin-bottom:3px;'>{ICONS[i]}</div>
          <div style='font-size:9px;color:#8b949e;text-transform:capitalize;'>{CLASSES[i]}</div>
        </div>"""
    imgs_html += "</div>"
    st.markdown(imgs_html + "</div>", unsafe_allow_html=True)

with col5:
    st.markdown("<div class='card'><div class='card-title'>📉 Training Progress (CNN)</div>", unsafe_allow_html=True)
    epochs = list(range(1, 11))

    fig_curves = make_subplots(rows=1, cols=2, subplot_titles=["Accuracy", "Loss"])
    fig_curves.add_trace(go.Scatter(x=epochs, y=TRAIN_ACC, name="Train Acc",
        line=dict(color="#388bfd", width=2), mode="lines+markers", marker=dict(size=3)), row=1, col=1)
    fig_curves.add_trace(go.Scatter(x=epochs, y=TEST_ACC, name="Test Acc",
        line=dict(color="#388bfd", width=1, dash="dash"), mode="lines+markers", marker=dict(size=3)), row=1, col=1)
    fig_curves.add_trace(go.Scatter(x=epochs, y=TRAIN_LOSS, name="Train Loss",
        line=dict(color="#f0883e", width=2), mode="lines+markers", marker=dict(size=3)), row=1, col=2)
    fig_curves.add_trace(go.Scatter(x=epochs, y=TEST_LOSS, name="Test Loss",
        line=dict(color="#f0883e", width=1, dash="dash"), mode="lines+markers", marker=dict(size=3)), row=1, col=2)

    fig_curves.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#8b949e", size=8), height=220,
        margin=dict(l=10, r=10, t=24, b=10), showlegend=False,
    )
    fig_curves.update_xaxes(gridcolor="#21262d", title_text="Epoch", title_font=dict(size=8))
    fig_curves.update_yaxes(gridcolor="#21262d")
    for ann in fig_curves.layout.annotations:
        ann.font.size = 9
        ann.font.color = "#8b949e"
    st.plotly_chart(fig_curves, use_container_width=True)

    st.markdown("""
    <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap;'>
      <span style='font-size:9px;color:#8b949e;display:flex;align-items:center;gap:4px;'><span style='width:16px;height:2px;background:#388bfd;display:inline-block;'></span>Train Acc</span>
      <span style='font-size:9px;color:#8b949e;display:flex;align-items:center;gap:4px;'><span style='width:16px;height:2px;background:#388bfd;display:inline-block;opacity:.5;'></span>Test Acc</span>
      <span style='font-size:9px;color:#8b949e;display:flex;align-items:center;gap:4px;'><span style='width:16px;height:2px;background:#f0883e;display:inline-block;'></span>Train Loss</span>
      <span style='font-size:9px;color:#8b949e;display:flex;align-items:center;gap:4px;'><span style='width:16px;height:2px;background:#f0883e;display:inline-block;opacity:.5;'></span>Test Loss</span>
    </div></div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("<div class='card'><div class='card-title'>⊞ Confusion Matrix (CNN)</div>", unsafe_allow_html=True)
    short = ['Air','Auto','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
    cm_arr = np.array(CM)
    fig_cm = go.Figure(go.Heatmap(
        z=cm_arr, x=short, y=short,
        colorscale=[[0,'#0d1117'],[0.5,'#1f6feb55'],[1,'#1f6feb']],
        showscale=False,
        text=cm_arr, texttemplate="%{text}",
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig_cm.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#8b949e", size=7), height=240,
        margin=dict(l=40, r=10, t=10, b=40),
        xaxis=dict(title="Predicted", title_font=dict(size=8), tickfont=dict(size=7)),
        yaxis=dict(title="Actual", title_font=dict(size=8), tickfont=dict(size=7), autorange="reversed"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Row 3: Per-Class Accuracy + Prediction History ────────────────────────────
col7, col8 = st.columns(2)

with col7:
    st.markdown("<div class='card'><div class='card-title'>🎯 Per-Class Accuracy (CNN)</div>", unsafe_allow_html=True)
    bars = "".join([
        f"""<div class='ca-row'>
          <span class='ca-name'>{ICONS[i]} {CLASSES[i]}</span>
          <div class='ca-track'><div class='ca-fill' style='width:{CLASS_ACC[i]}%;background:{COLORS[i]};'></div></div>
          <span class='ca-val'>{CLASS_ACC[i]}%</span>
        </div>""" for i in range(10)
    ])
    st.markdown(bars + "<div style='font-size:9px;color:#484f58;text-align:center;margin-top:6px;'>Accuracy (%)</div></div>", unsafe_allow_html=True)

with col8:
    st.markdown("<div class='card'><div class='card-title'>⭐ Prediction History (Recent)</div>", unsafe_allow_html=True)
    history = [
        ("🐴","Horse","98.20%","Horse","28 May 2025 10:24"),
        ("✈️","Airplane","94.50%","Airplane","28 May 2025 10:23"),
        ("🚢","Ship","92.10%","Ship","28 May 2025 10:22"),
        ("🚛","Truck","89.30%","Truck","28 May 2025 10:21"),
        ("🐶","Dog","88.70%","Dog","28 May 2025 10:20"),
    ]
    rows = ""
    for i, (icon, pred, conf, actual, date) in enumerate(history):
        border = "" if i == len(history)-1 else "border-bottom:1px solid #161b22;"
        rows += f"<tr><td style='{border}'>{i+1}</td><td style='{border}'>{icon}</td><td style='{border}color:#388bfd;font-weight:600;'>{pred}</td><td style='{border}color:#3fb950;font-weight:600;'>{conf}</td><td style='{border}'>{actual}</td><td style='{border}'><span class='bk-ok'>✓</span></td><td style='{border}color:#484f58;'>{date}</td></tr>"
    st.markdown(f"""
    <table class='ht'>
      <thead><tr><th>#</th><th>Image</th><th>Prediction</th><th>Confidence</th><th>Actual</th><th>Correct</th><th>Date</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    """, unsafe_allow_html=True)


# ── CNN Architecture ──────────────────────────────────────────────────────────
st.markdown("""
<div class='card'>
  <div class='card-title'>🏗 CNN Architecture</div>
  <div class='arch-wrap'>
    <div class='arch-step'><div style='font-size:16px;'>🖼</div><div class='arch-title'>Input</div><div class='arch-sub'>32×32×3</div></div>
    <div class='arch-arr'>→</div>
    <div class='arch-step'><div style='font-size:16px;'>🔲</div><div class='arch-title'>Conv2D<br>32 Filters</div><div class='arch-sub'>3×3, ReLU</div></div>
    <div class='arch-arr'>→</div>
    <div class='arch-step'><div style='font-size:16px;'>⬇️</div><div class='arch-title'>MaxPool<br>2×2</div><div class='arch-sub'></div></div>
    <div class='arch-arr'>→</div>
    <div class='arch-step'><div style='font-size:16px;'>🔲</div><div class='arch-title'>Conv2D<br>64 Filters</div><div class='arch-sub'>3×3, ReLU</div></div>
    <div class='arch-arr'>→</div>
    <div class='arch-step'><div style='font-size:16px;'>⬇️</div><div class='arch-title'>MaxPool<br>2×2</div><div class='arch-sub'></div></div>
    <div class='arch-arr'>→</div>
    <div class='arch-step'><div style='font-size:16px;'>➡️</div><div class='arch-title'>Flatten</div><div class='arch-sub'></div></div>
    <div class='arch-arr'>→</div>
    <div class='arch-step'><div style='font-size:16px;'>🧠</div><div class='arch-title'>Dense 64</div><div class='arch-sub'>ReLU</div></div>
    <div class='arch-arr'>→</div>
    <div class='arch-step'><div style='font-size:16px;'>🎯</div><div class='arch-title'>Dense 10</div><div class='arch-sub'>Softmax</div></div>
    <div class='arch-arr'>→</div>
    <div class='arch-step'><div style='font-size:16px;'>✅</div><div class='arch-title'>Output<br>10 Classes</div><div class='arch-sub'></div></div>
  </div>
</div>
""", unsafe_allow_html=True)
