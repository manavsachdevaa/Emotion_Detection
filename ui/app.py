import streamlit as st
import cv2
import sys
import os
import time
import numpy as np
import plotly.graph_objects as go

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
    "Happy":     "#FBBF24",
    "Sad":       "#60A5FA",
    "Angry":     "#F87171",
    "Surprised": "#34D399",
    "Neutral":   "#94A3B8",
    "Fearful":   "#C084FC",
    "Disgusted": "#FB923C",
}

EMOTION_GLOW = {
    "Happy":     "#FBBF2455",
    "Sad":       "#60A5FA55",
    "Angry":     "#F8717155",
    "Surprised": "#34D39955",
    "Neutral":   "#94A3B855",
    "Fearful":   "#C084FC55",
    "Disgusted": "#FB923C55",
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
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #070b14;
    color: #e2e8f0;
  }

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Subtle grid background */
  .stApp {
    background-color: #070b14;
    background-image:
      linear-gradient(rgba(99, 102, 241, 0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(99, 102, 241, 0.04) 1px, transparent 1px);
    background-size: 40px 40px;
  }

  /* Top banner */
  .banner {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 24px 0 16px 0;
    margin-bottom: 28px;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2);
  }
  .banner-icon-wrap {
    width: 52px; height: 52px;
    border-radius: 14px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.45);
  }
  .banner-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
  }
  .banner-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 3px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
  }
  .banner-badge {
    margin-left: auto;
    padding: 6px 14px;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    font-size: 0.72rem;
    color: #a78bfa;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.6px;
    white-space: nowrap;
  }

  /* Glass card */
  .glass-card {
    background: rgba(15, 20, 35, 0.7);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 16px;
    backdrop-filter: blur(12px);
    padding: 20px;
  }

  /* Metric card */
  .metric-card {
    background: rgba(15, 20, 35, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 14px;
    padding: 16px 12px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover {
    border-color: rgba(167, 139, 250, 0.45);
  }
  .metric-label {
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.45rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 4px;
  }

  /* Section heading */
  .section-title {
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(99, 102, 241, 0.15);
  }

  /* Divider */
  .divider { border-top: 1px solid rgba(99, 102, 241, 0.12); margin: 20px 0; }

  /* Live status */
  .status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 8px #34d399, 0 0 16px #34d39944;
    margin-right: 7px;
    animation: pulse 1.4s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px #34d399, 0 0 16px #34d39944; }
    50%       { opacity: 0.5; box-shadow: 0 0 4px #34d399; }
  }
  .status-text {
    font-size: 0.76rem;
    color: #34d399;
    font-family: 'Space Mono', monospace;
    vertical-align: middle;
    letter-spacing: 0.6px;
  }
  .status-idle {
    font-size: 0.76rem;
    color: #475569;
    font-family: 'Space Mono', monospace;
  }

  /* Idle placeholder */
  .idle-box {
    background: rgba(15, 20, 35, 0.7);
    border: 1px dashed rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 360px;
    gap: 14px;
  }
  .idle-icon { font-size: 3.2rem; filter: grayscale(30%); }
  .idle-text {
    font-family: 'Space Mono', monospace;
    color: #334155;
    font-size: 0.82rem;
    letter-spacing: 0.4px;
  }

  /* Plotly chart */
  .js-plotly-plot .plotly { background: transparent !important; }

  /* Streamlit toggle styling */
  .stToggle > label {
    color: #a78bfa !important;
    font-weight: 500;
  }
</style>
""", unsafe_allow_html=True)

# ================= BANNER =================
st.markdown("""
<div class="banner">
  <div class="banner-icon-wrap">🧠</div>
  <div>
    <div class="banner-title">Emotion Detector</div>
    <div class="banner-sub">Hybrid Model · Real-Time · Computer Vision</div>
  </div>
  <div class="banner-badge">⚡ Live Analysis</div>
</div>
""", unsafe_allow_html=True)

# ================= LAYOUT =================
left_col, right_col = st.columns([3, 2], gap="large")

# ---- LEFT: Camera feed ----
with left_col:
    st.markdown('<div class="section-title">Live Feed</div>', unsafe_allow_html=True)
    frame_window = st.empty()
    status_area  = st.empty()
    run          = st.toggle("▶  Start Camera", value=False)

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
            opacity=0.85,
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#64748b", size=11, family="Space Mono"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=55, t=4, b=4),
        height=290,
        xaxis=dict(
            range=[0, 110],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#94a3b8", size=12, family="Inter"),
            autorange="reversed",
        ),
        bargap=0.3,
        font=dict(color="#e2e8f0"),
    )
    return fig


def render_top_emotion(name, pct, color, glow):
    emoji = emotion_emoji(name)
    return f"""
    <div style="text-align:center; padding: 18px 0 8px 0;">
      <div style="font-size:3.2rem; line-height:1; filter: drop-shadow(0 0 12px {color}88);">{emoji}</div>
      <div style="font-family:'Space Mono',monospace; font-size:1.4rem; font-weight:700;
                  color:{color}; margin-top:12px; text-shadow: 0 0 20px {color}66;">{name}</div>
      <div style="display:inline-block; margin-top:10px; padding:5px 16px;
                  border-radius:999px; background:{glow};
                  border:1px solid {color}55; font-size:0.82rem; color:{color}; font-weight:600;">
        {pct:.1f}% confidence
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

        # Sort by score descending
        scores    = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        top_name  = list(scores.keys())[0]
        top_pct   = list(scores.values())[0] * 100
        top_color = EMOTION_COLORS.get(top_name, "#a78bfa")
        top_glow  = EMOTION_GLOW.get(top_name, "#a78bfa33")

        # ── Update UI ──────────────────────────────
        frame_window.image(processed_frame, channels="BGR", use_container_width=True)

        top_emotion_area.markdown(
            render_top_emotion(top_name, top_pct, top_color, top_glow),
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
    <div class="idle-box">
      <div class="idle-icon">📷</div>
      <div class="idle-text">Toggle the switch to start</div>
    </div>
    """, unsafe_allow_html=True)

    status_area.markdown(
        '<span class="status-idle">● IDLE</span>',
        unsafe_allow_html=True,
    )

# ================= CLEANUP =================
cap.release()