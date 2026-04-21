import streamlit as st
import cv2
import sys
import os
import time
import numpy as np
import plotly.graph_objects as go
import random  # Remove this if your model provides real scores

# ================= FIX IMPORT PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, PARENT_DIR)

from core.detector import process_frame

# ===============================================================
# NOTE: Replace `get_emotion_scores(frame)` with your real model.
# It should return a dict like:
#   {"Happy": 0.72, "Sad": 0.05, "Angry": 0.03, ...}  (values 0–1)
# ===============================================================
def get_emotion_scores(frame):
    """STUB — replace with your hybrid model's probability output."""
    emotions = ["Happy", "Sad", "Angry", "Surprised", "Neutral", "Fearful", "Disgusted"]
    scores = np.random.dirichlet(np.ones(len(emotions)))  # fake softmax
    return dict(zip(emotions, scores))

# ================= EMOTION PALETTE =================
EMOTION_COLORS = {
    "Happy":     "#FFD166",
    "Sad":       "#118AB2",
    "Angry":     "#EF476F",
    "Surprised": "#06D6A0",
    "Neutral":   "#A8DADC",
    "Fearful":   "#9B5DE5",
    "Disgusted": "#F4845F",
}

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ================= GLOBAL CSS =================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8eaf0;
  }

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Top banner */
  .banner {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 20px 0 8px 0;
    border-bottom: 1px solid #1e2130;
    margin-bottom: 24px;
  }
  .banner-icon {
    font-size: 2.4rem;
    line-height: 1;
  }
  .banner-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
  }
  .banner-sub {
    font-size: 0.82rem;
    color: #6b7280;
    margin-top: 2px;
    letter-spacing: 0.4px;
    text-transform: uppercase;
  }

  /* Metric pill */
  .metric-card {
    background: #161922;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-label {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f9fafb;
    margin-top: 4px;
  }
  .metric-accent { color: #FFD166; }

  /* Emotion badge */
  .emotion-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-top: 8px;
    background: #FFD16622;
    color: #FFD166;
    border: 1px solid #FFD16655;
  }

  /* Section heading */
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 10px;
  }

  /* Divider */
  .divider { border-top: 1px solid #1e2130; margin: 18px 0; }

  /* Status dot */
  .status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 6px #10b981;
    margin-right: 6px;
    animation: pulse 1.4s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.35; }
  }
  .status-text {
    font-size: 0.78rem;
    color: #10b981;
    font-family: 'Space Mono', monospace;
    vertical-align: middle;
  }

  /* Plotly chart background */
  .js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ================= BANNER =================
st.markdown("""
<div class="banner">
  <div class="banner-icon">🧠</div>
  <div>
    <div class="banner-title">Emotion Detector</div>
    <div class="banner-sub">Hybrid Model · Real-Time · Computer Vision</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ================= LAYOUT =================
left_col, right_col = st.columns([3, 2], gap="large")

# ---- LEFT: Camera feed ----
with left_col:
    st.markdown('<div class="section-title">Live Feed</div>', unsafe_allow_html=True)
    frame_window   = st.empty()
    status_area    = st.empty()
    run            = st.toggle("▶  Start Camera", value=False)

# ---- RIGHT: Analytics panel ----
with right_col:
    st.markdown('<div class="section-title">Emotion Analysis</div>', unsafe_allow_html=True)

    # Dominant emotion + confidence
    top_emotion_area = st.empty()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Metric row
    m1, m2, m3 = st.columns(3)
    fps_area  = m1.empty()
    conf_area = m2.empty()
    frm_area  = m3.empty()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Distribution</div>', unsafe_allow_html=True)
    chart_area = st.empty()

# ================= HELPERS =================
def render_bar_chart(scores: dict):
    emotions = list(scores.keys())
    values   = [round(v * 100, 1) for v in scores.values()]
    colors   = [EMOTION_COLORS.get(e, "#888") for e in emotions]

    fig = go.Figure(go.Bar(
        x=values,
        y=emotions,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#9ca3af", size=11, family="Space Mono"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=50, t=4, b=4),
        height=280,
        xaxis=dict(
            range=[0, 105],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#d1d5db", size=12, family="DM Sans"),
            autorange="reversed",
        ),
        bargap=0.28,
        font=dict(color="#e8eaf0"),
    )
    return fig


def render_top_emotion(name, pct, color):
    return f"""
    <div style="text-align:center; padding: 14px 0 6px 0;">
      <div style="font-size:2.6rem; line-height:1.1;">{emotion_emoji(name)}</div>
      <div style="font-family:'Space Mono',monospace; font-size:1.35rem; font-weight:700;
                  color:{color}; margin-top:8px;">{name}</div>
      <div style="font-size:0.82rem; color:#6b7280; margin-top:4px;">
        Confidence: <span style="color:{color}; font-weight:600;">{pct:.1f}%</span>
      </div>
    </div>
    """


def render_metric(label, value):
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
    </div>
    """


def emotion_emoji(name):
    return {
        "Happy": "😊", "Sad": "😢", "Angry": "😠",
        "Surprised": "😲", "Neutral": "😐",
        "Fearful": "😨", "Disgusted": "🤢",
    }.get(name, "🤔")


# ================= CAMERA + LOOP =================
cap = cv2.VideoCapture(0)

if run:
    status_area.markdown(
        '<span class="status-dot"></span><span class="status-text">LIVE</span>',
        unsafe_allow_html=True,
    )
    frame_count = 0
    t_start = time.time()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Camera not accessible.")
            break

        frame_count += 1
        t0 = time.time()

        # ── Inference ──────────────────────────────
        processed_frame = process_frame(frame)
        scores          = get_emotion_scores(frame)   # ← plug in your real scores
        # ───────────────────────────────────────────

        elapsed = time.time() - t0
        fps     = 1.0 / elapsed if elapsed > 0 else 0
        total_s = time.time() - t_start

        # Sort by score descending
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        top_name = list(scores.keys())[0]
        top_pct  = list(scores.values())[0] * 100
        top_color = EMOTION_COLORS.get(top_name, "#FFD166")

        # ── Update UI ──────────────────────────────
        frame_window.image(processed_frame, channels="BGR", use_container_width=True)

        top_emotion_area.markdown(
            render_top_emotion(top_name, top_pct, top_color),
            unsafe_allow_html=True,
        )

        fps_area.markdown(render_metric("FPS", f"{fps:.0f}"), unsafe_allow_html=True)
        conf_area.markdown(render_metric("Confidence", f"{top_pct:.0f}%"), unsafe_allow_html=True)
        frm_area.markdown(render_metric("Frames", frame_count), unsafe_allow_html=True)

        chart_area.plotly_chart(
            render_bar_chart(scores),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        time.sleep(0.03)

else:
    # Idle state
    frame_window.markdown("""
    <div style="background:#161922; border:1px solid #1e2130; border-radius:16px;
                display:flex; flex-direction:column; align-items:center;
                justify-content:center; height:360px; gap:12px;">
      <div style="font-size:3rem;">📷</div>
      <div style="font-family:'Space Mono',monospace; color:#4b5563; font-size:0.85rem;">
        Toggle the switch to start
      </div>
    </div>
    """, unsafe_allow_html=True)

    status_area.markdown(
        '<span style="font-size:0.78rem; color:#4b5563; font-family:\'Space Mono\',monospace;">● IDLE</span>',
        unsafe_allow_html=True,
    )

# ================= CLEANUP =================
cap.release()