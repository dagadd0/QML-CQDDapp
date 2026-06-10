import streamlit as st
import numpy as np

from quantum_backend_final import (
    run_simulation,
    create_bloch_plotly
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="CQDD · Quantum State Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# CONSTANTS
# =====================================================

PARAM_FILES = {
    "Polar Points": "params_clusters.npy",
    "Rotated Rings": "params_rings.npy",
    "Parallel Rings": "params_ringsparalel.npy",
}

MODEL_CONFIG = {
    "Polar Points": {"nancilla": 2, "ndataq": 1},
    "Rotated Rings": {"nancilla": 3, "ndataq": 1},
    "Parallel Rings": {"nancilla": 3, "ndataq": 1},
}

# =====================================================
# CUSTOM CSS — Dark Scientific Terminal
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@300;400;500;600&family=Instrument+Sans:wght@300;400;500&display=swap');

:root {
    --bg:       #080c10;
    --bg-card:  #0d1117;
    --bg-hover: #121820;
    --border:   #1c2a38;
    --border-hi:#1e4060;
    --cyan:     #00d4ff;
    --cyan-dim: #0097b8;
    --cyan-glow:rgba(0,212,255,0.12);
    --amber:    #f0a830;
    --green:    #39d07e;
    --red:      #ff4f6a;
    --text:     #c8d8e8;
    --text-dim: #5a7080;
    --text-hi:  #eaf4ff;
    --mono:     'JetBrains Mono', monospace;
    --sans:     'Instrument Sans', sans-serif;
    --serif:    'DM Serif Display', serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1300px !important;
}

.cqdd-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.2rem;
    margin-bottom: 2.2rem;
}
.cqdd-header .label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 0.35rem;
}
.cqdd-header h1 {
    font-family: var(--serif) !important;
    font-size: 2.1rem !important;
    color: var(--text-hi) !important;
    margin: 0 0 0.4rem !important;
    font-weight: 400 !important;
    letter-spacing: -0.01em;
}
.cqdd-header p {
    font-size: 0.82rem;
    color: var(--text-dim);
    margin: 0;
    font-family: var(--mono);
}

.section-title {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--cyan-dim);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

.stat-row {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 160px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: var(--border-hi); }
.stat-card .sc-label {
    font-family: var(--mono);
    font-size: 0.57rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.3rem;
}
.stat-card .sc-value {
    font-family: var(--mono);
    font-size: 1.15rem;
    color: var(--cyan);
    font-weight: 500;
}
.stat-card .sc-unit {
    font-size: 0.65rem;
    color: var(--text-dim);
    margin-left: 0.2em;
}

.panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
}
.panel-glow {
    box-shadow: 0 0 40px var(--cyan-glow);
}

.streamlit-expanderHeader {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    color: var(--text-dim) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    font-family: var(--mono) !important;
    font-size: 0.77rem !important;
    color: var(--text) !important;
    line-height: 1.8 !important;
}

.stButton > button {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    border-radius: 4px !important;
    padding: 0.55rem 1.8rem !important;
    transition: background 0.2s, box-shadow 0.2s !important;
}
.stButton > button:hover {
    background: var(--cyan-glow) !important;
    box-shadow: 0 0 16px var(--cyan-glow) !important;
}
.stButton > button[kind="primary"] {
    background: var(--cyan) !important;
    color: #040810 !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #33ddff !important;
    box-shadow: 0 0 24px rgba(0,212,255,0.3) !important;
}

.stDownloadButton > button {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid var(--border-hi) !important;
    color: var(--text-dim) !important;
    border-radius: 4px !important;
}
.stDownloadButton > button:hover {
    border-color: var(--cyan-dim) !important;
    color: var(--cyan) !important;
}

section[data-testid="stSidebar"] {
    background: #090d12 !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--mono) !important;
    font-size: 0.58rem !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    color: var(--cyan-dim) !important;
}
section[data-testid="stSidebar"] label {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    color: var(--text) !important;
    letter-spacing: 0.05em !important;
}

[data-baseweb="slider"] [data-testid="stTickBar"] { display: none; }
[data-baseweb="slider"] div[role="slider"] {
    background: var(--cyan) !important;
    border: none !important;
    box-shadow: 0 0 8px var(--cyan-glow) !important;
}

[data-baseweb="select"] div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
}

.stRadio label {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    color: var(--text) !important;
}

.stTextInput input, .stNumberInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--cyan) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    border-radius: 4px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--cyan-dim) !important;
    box-shadow: 0 0 0 2px var(--cyan-glow) !important;
}

.stAlert {
    background: var(--bg-card) !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.73rem !important;
}
div[data-testid="stInfoMessage"] {
    border-left: 3px solid var(--cyan-dim) !important;
}
div[data-testid="stWarningMessage"] {
    border-left: 3px solid var(--amber) !important;
}

.stSpinner > div {
    border-top-color: var(--cyan) !important;
}

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

.shape-badge {
    display: inline-flex;
    gap: 0.4rem;
    align-items: center;
    background: var(--bg);
    border: 1px solid var(--border-hi);
    border-radius: 4px;
    padding: 0.3rem 0.8rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--cyan);
    margin-top: 0.4rem;
}
.shape-badge .arrow { color: var(--text-dim); }

</style>
""", unsafe_allow_html=True)

# =====================================================
# HELPERS
# =====================================================

@st.cache_data(show_spinner=False)
def load_params_cached(params_file: str) -> np.ndarray:
    return np.load(params_file, allow_pickle=False)

@st.cache_data(show_spinner=False)
def cached_simulation(distribution, mu1, mu2, nstates):
    return run_simulation(distribution, mu1, mu2, nstates)

def parse_pi(value: str) -> float:
    try:
        value = value.replace("π", "pi")
        return float(eval(value, {"pi": np.pi}))
    except Exception:
        return float(value)

def fmt_angle(v):
    ratio = v / np.pi
    if abs(ratio - round(ratio)) < 1e-9:
        n = int(round(ratio))
        return f"{n}π" if n != 0 else "0"
    for denom in [2, 3, 4, 6, 8]:
        num = ratio * denom
        if abs(num - round(num)) < 1e-9:
            n = int(round(num))
            return f"{n}π/{denom}"
    return f"{v:.4f} rad"

def fmt_conditioning_value(distribution, value):
    if distribution == "Parallel Rings":
        return f"{value:.3f}"
    return fmt_angle(value)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="cqdd-header">
  <div class="label">Conditioned Quantum Denoising Diffusion</div>
  <h1>CQDD &mdash; Quantum State Generator</h1>
  <p>PQC-based trained conditioned quantum denoising diffusion model &nbsp;·&nbsp; Bloch sphere visualization</p>
  <p>
    <a href="https://github.com/dagadd0/QML-CQDDmodel" target="_blank" style="color:#00d4ff">
      GitHub: CQDD Model
    </a>
    &nbsp;|&nbsp;
    <a href="https://github.com/dagadd0/QML-CQDDapp" target="_blank" style="color:#00d4ff">
      GitHub: CQDD App
    </a>
  </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("### Simulation")

    nstates = st.slider("Number of states", 10, 1000, 500, step=10)

    distribution = st.selectbox(
        "Target distribution",
        ["Polar Points", "Rotated Rings", "Parallel Rings"]
    )

    params_file = PARAM_FILES[distribution]
    params = load_params_cached(params_file)

    if params.ndim != 3:
        raise ValueError(f"{params_file} must have shape (T, L, P). Got {params.shape}")

    num_steps, num_layers, param_dim = params.shape

    num_ancillas = MODEL_CONFIG[distribution]["nancilla"]
    num_dataq = MODEL_CONFIG[distribution]["ndataq"]
    num_qubits = num_ancillas + num_dataq

    expected_param_dim = 2 * num_qubits
    if param_dim != expected_param_dim:
        raise ValueError(
            f"Shape mismatch for '{distribution}': expected last dimension "
            f"{expected_param_dim} (= 2 * {num_qubits}), got {param_dim}. "
            f"File: {params_file}, shape: {params.shape}"
        )

    st.markdown("---")
    st.markdown("### Conditioning parameters")

    if distribution == "Parallel Rings" or distribution == "Rotated Rings":
        mu1_mode = st.radio("μ₁ input", ["Slider", "Manual"], horizontal=True)

        if mu1_mode == "Slider":
            mu1 = st.slider(
                "μ₁",
                0.0,
                np.pi,
                0.0,
                step=0.01,
                key="mu1_sl_parallel",
            )
        else:
            mu1 = st.number_input(
                "μ₁",
                min_value=0.0,
                max_value=np.pi,
                value=0.0,
                step=0.01,
                format="%.3f",
                key="mu1_num_parallel",
            )
        mu2 = 0.0

    else:
        mu1_mode = st.radio("μ₁ input", ["Slider", "Manual"], horizontal=True)
        if mu1_mode == "Slider":
            mu1 = st.slider("μ₁", 0.0, float(2 * np.pi), float(np.pi / 2), key="mu1_sl")
        else:
            mu1 = parse_pi(st.text_input("μ₁  (e.g. pi/2)", "pi/2", key="mu1_tx"))

        st.markdown("---")

        mu2_mode = st.radio("μ₂ input", ["Slider", "Manual"], horizontal=True)
        if mu2_mode == "Slider":
            mu2 = st.slider("μ₂", 0.0, float(2 * np.pi), 0.0, key="mu2_sl")
        else:
            mu2 = parse_pi(st.text_input("μ₂ (e.g. 2*pi)", "5*pi/2", key="mu2_tx"))

# Reset when configuration changes
current_signature = (distribution, params_file, nstates, mu1, mu2, num_steps, num_layers)
if st.session_state.get("ui_signature") != current_signature:
    st.session_state.final_states = None
    st.session_state.ui_signature = current_signature

if "final_states" not in st.session_state:
    st.session_state.final_states = None

# =====================================================
# STAT CARDS
# =====================================================

st.markdown(
    f"""
<div class="stat-row">
  <div class="stat-card">
    <div class="sc-label">Data qubits</div>
    <div class="sc-value">{num_dataq}<span class="sc-unit">qubit</span></div>
  </div>
  <div class="stat-card">
    <div class="sc-label">Ancillas</div>
    <div class="sc-value">{num_ancillas}<span class="sc-unit">qubits</span></div>
  </div>
  <div class="stat-card">
    <div class="sc-label">Total qubits</div>
    <div class="sc-value">{num_qubits}<span class="sc-unit">qubits</span></div>
  </div>
  <div class="stat-card">
    <div class="sc-label">Denoising steps</div>
    <div class="sc-value">T<span class="sc-unit">= {num_steps}</span></div>
  </div>
  <div class="stat-card">
    <div class="sc-label">PQC layers</div>
    <div class="sc-value">L<span class="sc-unit">= {num_layers}</span></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =====================================================
# RUN BUTTON
# =====================================================

run_col, _ = st.columns([1, 4])
with run_col:
    run = st.button("▶  Run simulation", type="primary", use_container_width=True)

if run:
    with st.spinner("Evolving quantum circuit…"):
        st.session_state.final_states = cached_simulation(
            distribution,
            mu1,
            mu2,
            nstates
        )

# =====================================================
# MAIN LAYOUT
# =====================================================

col1, col2 = st.columns([1.3, 1], gap="large")

with col1:
    st.markdown('<div class="section-title">Bloch sphere</div>', unsafe_allow_html=True)

    if st.session_state.final_states is not None:
        fig = create_bloch_plotly(st.session_state.final_states)

        fig.update_layout(
            paper_bgcolor="rgba(13,17,23,0)",
            plot_bgcolor="rgba(13,17,23,0)",
            font=dict(family="JetBrains Mono, monospace", color="#c8d8e8", size=11),
            margin=dict(l=0, r=0, t=10, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f'<div style="font-family:var(--mono,monospace);font-size:0.65rem;'
            f'color:#5a7080;text-align:right;margin-top:-0.5rem">'
            f'{nstates} states rendered &nbsp;·&nbsp; μ₁ = {fmt_conditioning_value(distribution, mu1)}'
            f' &nbsp;·&nbsp; μ₂ = {fmt_conditioning_value(distribution, mu2)}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
<div style="
    background:#0d1117;
    border:1px dashed #1c2a38;
    border-radius:8px;
    height:380px;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    gap:0.8rem;
">
  <span style="font-size:2rem;opacity:0.2">⊙</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
               letter-spacing:0.12em;color:#3a5060;text-transform:uppercase">
    Awaiting simulation run
  </span>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-title">Variational quantum circuit (PQC)</div>', unsafe_allow_html=True)

    try:
        st.image("circuit.png", use_container_width=True)
    except:
        st.info("Circuit diagram not available")

    st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)

    with st.expander("Model details  ↗", expanded=False):
        st.markdown(
            f"""
**System configuration**

| Parameter | Value |
|---|---|
| Target distribution | {distribution} |
| Data qubit | {num_dataq} |
| Ancillas | {num_ancillas} |
| Total qubits | {num_qubits} |
| Denoising steps | T = {num_steps} |
| PQC layers | L = {num_layers} |
| Param tensor | {params.shape} |

---

**Circuit description**

A parameterized quantum circuit (PQC) evolves a random distribution of data qubits toward the selected target distribution.

At each timestep *t*:

1. Ancillas are initialized and rotated via μ₁ and μ₂  
2. A variational block of L = {num_layers} layers is applied (RX → RY → CZ entanglement)  
3. Ancillas are measured and discarded  
4. The system evolves toward the target distribution
"""
        )

    st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Trained parameters</div>', unsafe_allow_html=True)

    shape_str = " &nbsp;<span style='color:#5a7080'>×</span>&nbsp; ".join(
        f"<span style='color:#00d4ff'>{d}</span>" for d in params.shape
    )
    st.markdown(
        f'<div class="shape-badge">{params_file} &nbsp;'
        f'<span class="arrow">→</span>&nbsp; {shape_str}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="margin-top:0.8rem"></div>', unsafe_allow_html=True)

    with open(params_file, "rb") as f:
        params_bytes = f.read()

    st.download_button(
        label=f"↓  Download {params_file}",
        data=params_bytes,
        file_name=params_file,
        mime="application/octet-stream",
        use_container_width=True,
    )