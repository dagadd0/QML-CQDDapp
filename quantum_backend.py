import numpy as np
import plotly.graph_objects as go
from functools import lru_cache

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PARAM_FILES = {
    "Cluster States": "params_clusters.npy",
    "Rings":          "params_rings.npy",
    "Parallel Rings": "params_ringsparalel.npy",
}

MODEL_CONFIG = {
    "Cluster States": {"nancilla": 2, "ndataq": 1},
    "Rings":          {"nancilla": 3, "ndataq": 1},
    "Parallel Rings": {"nancilla": 3, "ndataq": 1},
}

# ─────────────────────────────────────────────
# PUERTAS CUÁNTICAS
# ─────────────────────────────────────────────

def rx(t): 
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex64)

def ry(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -s], [s, c]], dtype=np.complex64)

def rz(t):
    return np.array([[np.exp(-1j*t/2), 0], [0, np.exp(1j*t/2)]], dtype=np.complex64)

CZ = np.diag([1, 1, 1, -1]).astype(np.complex64)

# ─────────────────────────────────────────────
# OPERACIONES EN BATCH (todos los estados a la vez)
# ─────────────────────────────────────────────

def apply_1q_gate_batch(states, gate, target, nqubits):
    """
    Aplica una puerta de 1 qubit a todos los estados a la vez.

    states : (batch, 2**nqubits)
    gate   : (2, 2)
    returns: (batch, 2**nqubits)
    """
    B = states.shape[0]
    # Reshape a (batch, 2, 2, ..., 2)
    t = states.reshape((B,) + (2,)*nqubits)

    # Lleva el qubit target al eje 1 (tras batch)
    axes = list(range(nqubits + 1))
    axes[1], axes[target + 1] = axes[target + 1], axes[1]
    t = t.transpose(axes)

    # (batch, 2, rest) → aplica gate
    t = t.reshape(B, 2, -1)
    t = np.einsum('ij,bjk->bik', gate, t)

    # Deshace la transposición
    t = t.reshape((B,) + (2,)*nqubits)
    inv = [0] * (nqubits + 1)
    for i, a in enumerate(axes):
        inv[a] = i
    t = t.transpose(inv)

    return t.reshape(B, -1)


def apply_2q_gate_batch(states, gate, q1, q2, nqubits):
    """
    Aplica una puerta de 2 qubits a todos los estados a la vez.

    states : (batch, 2**nqubits)
    gate   : (4, 4)
    returns: (batch, 2**nqubits)
    """
    B = states.shape[0]
    if q1 > q2:
        q1, q2 = q2, q1

    t = states.reshape((B,) + (2,)*nqubits)

    axes = list(range(nqubits + 1))
    axes[1], axes[q1 + 1] = axes[q1 + 1], axes[1]
    axes[2], axes[q2 + 1] = axes[q2 + 1], axes[2]
    t = t.transpose(axes)

    t = t.reshape(B, 4, -1)
    t = np.einsum('ij,bjk->bik', gate, t)

    t = t.reshape((B,) + (2,)*nqubits)
    inv = [0] * (nqubits + 1)
    for i, a in enumerate(axes):
        inv[a] = i
    t = t.transpose(inv)

    return t.reshape(B, -1)


# ─────────────────────────────────────────────
# ANCILLAS
# ─────────────────────────────────────────────

def add_ancillas_batch(states, nancilla):
    """
    Añade |0⟩^⊗nancilla a cada estado de datos.

    states  : (batch, 2**ndataq)
    returns : (batch, 2**nqubits)
    """
    B, D = states.shape
    ancilla = np.zeros(2**nancilla, dtype=np.complex64)
    ancilla[0] = 1.0
    # kron fila a fila con broadcasting
    # outer: (B, D, 2**na) → reshape (B, D * 2**na)
    return (states[:, :, None] * ancilla[None, None, :]).reshape(B, -1)


def measure_ancillas_batch(states, nancilla, ndataq, rng):
    """
    Medición estocástica de ancillas para cada estado.
    Cada estado recibe un resultado de medición diferente (aleatorio),
    lo que produce la distribución de salida.

    states  : (batch, 2**nqubits)
    returns : (batch, 2**ndataq)
    """
    B = states.shape[0]
    D  = 2**ndataq
    NA = 2**nancilla

    # Reshape: (batch, data_dim, ancilla_dim)
    s = states.reshape(B, D, NA)

    # Probabilidad de cada resultado de ancilla para cada estado
    probs = np.sum(np.abs(s)**2, axis=1)          # (B, NA)
    probs /= probs.sum(axis=1, keepdims=True) + 1e-10

    # Muestrea un resultado por estado (¡diferente para cada uno!)
    outcomes = np.array([
        rng.choice(NA, p=probs[i])
        for i in range(B)
    ])                                             # (B,)

    # Estado de datos tras la medición
    post = s[np.arange(B), :, outcomes]            # (B, D)
    post /= np.linalg.norm(post, axis=1, keepdims=True) + 1e-12

    return post.astype(np.complex64)


# ─────────────────────────────────────────────
# CIRCUITO PARAMETRIZADO (batch)
# ─────────────────────────────────────────────

def apply_pqc_batch(states, params, nqubits, nlayers):
    """
    Aplica nlayers capas del PQC a todos los estados en batch.

    states  : (batch, 2**nqubits)
    params  : (nlayers, 2*nqubits)
    returns : (batch, 2**nqubits)
    """
    for l in range(nlayers):
        angles = params[l]
        rx_a = angles[0::2]   # ángulos RX
        ry_a = angles[1::2]   # ángulos RY

        for q in range(nqubits):
            states = apply_1q_gate_batch(states, rx(rx_a[q]), q, nqubits)
        for q in range(nqubits):
            states = apply_1q_gate_batch(states, ry(ry_a[q]), q, nqubits)
        for q in range(nqubits - 1):
            states = apply_2q_gate_batch(states, CZ, q, q+1, nqubits)

    return states


def encode_mu_batch(states, mu1, mu2, nancilla, ndataq, nqubits):
    """
    Aplica RY(μ₁) + RZ(μ₂) a cada ancilla sobre todos los estados.

    states : (batch, 2**nqubits)
    """
    ry_gate = ry(float(mu1))
    rz_gate = rz(float(mu2))

    for i in range(nancilla):
        q = ndataq + i
        states = apply_1q_gate_batch(states, ry_gate, q, nqubits)
        states = apply_1q_gate_batch(states, rz_gate, q, nqubits)

    return states


# ─────────────────────────────────────────────
# CARGA DE PARÁMETROS
# ─────────────────────────────────────────────

@lru_cache(maxsize=3)
def load_params(distribution):
    return np.load(PARAM_FILES[distribution], allow_pickle=False)


# ─────────────────────────────────────────────
# ESTADOS HAAR-RANDOM
# ─────────────────────────────────────────────

def haar_random_states(N, seed=0):
    """
    Genera N estados de 1 qubit uniformemente distribuidos en la esfera de Bloch.
    """
    rng = np.random.default_rng(seed)
    u   = rng.uniform(0, 1, N)
    phi = rng.uniform(0, 2*np.pi, N)

    theta = np.arccos(1 - 2*u)
    alpha = np.cos(theta/2).astype(np.complex64)
    beta  = (np.exp(1j*phi) * np.sin(theta/2)).astype(np.complex64)

    return np.stack([alpha, beta], axis=1)   # (N, 2)


# ─────────────────────────────────────────────
# SIMULACIÓN PRINCIPAL
# ─────────────────────────────────────────────

def run_simulation(distribution, mu1, mu2, nstates):
    """
    Ejecuta la simulación CQDD completa de forma vectorizada.

    Args:
        distribution : "Cluster States" | "Rings" | "Parallel Rings"
        mu1, mu2     : parámetros de condicionamiento
        nstates      : número de estados

    Returns:
        states : (nstates, 2)  estados finales en la esfera de Bloch
    """
    cfg      = MODEL_CONFIG[distribution]
    nancilla = cfg["nancilla"]
    ndataq   = cfg["ndataq"]
    nqubits  = nancilla + ndataq

    if distribution == "Parallel Rings":
        mu2 = 0.0

    params = load_params(distribution)
    T, L, P = params.shape

    if P != 2 * nqubits:
        raise ValueError(
            f"Shape mismatch '{distribution}': esperado P={2*nqubits}, got {P}"
        )

    # Estados iniciales Haar-random
    states = haar_random_states(nstates, seed=42)   # (N, 2)

    # RNG para mediciones (seed fijo → resultados reproducibles)
    rng = np.random.default_rng(0)

    for t in range(T):
        # 1) Añade ancillas |0⟩ a cada estado de datos
        full = add_ancillas_batch(states, nancilla)                # (N, 2**nqubits)

        # 2) Encoding de μ en las ancillas
        full = encode_mu_batch(full, mu1, mu2, nancilla, ndataq, nqubits)

        # 3) Circuito parametrizado
        full = apply_pqc_batch(full, params[t], nqubits, L)        # (N, 2**nqubits)

        # 4) Medición estocástica de ancillas → vuelve a estados de 1 qubit
        states = measure_ancillas_batch(full, nancilla, ndataq, rng)  # (N, 2)

    return states


# ─────────────────────────────────────────────
# COORDENADAS DE BLOCH
# ─────────────────────────────────────────────

def states_to_bloch(states):
    """
    |ψ⟩ = α|0⟩ + β|1⟩  →  (x, y, z)

    x = 2 Re(α* β)
    y = 2 Im(α* β)
    z = |α|² − |β|²
    """
    alpha = states[:, 0]
    beta  = states[:, 1]
    ab    = np.conj(alpha) * beta

    return (
        2 * np.real(ab).astype(float),
        2 * np.imag(ab).astype(float),
        (np.abs(alpha)**2 - np.abs(beta)**2).astype(float),
    )


# ─────────────────────────────────────────────
# VISUALIZACIÓN PLOTLY
# ─────────────────────────────────────────────

def create_bloch_plotly(states):
    xs, ys, zs = states_to_bloch(states)

    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 30)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()

    fig.add_surface(
        x=sx, y=sy, z=sz,
        opacity=0.1, colorscale='Greys',
        showscale=False, hoverinfo='skip',
        name='Bloch sphere'
    )

    fig.add_scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=4, color='#00d4ff', opacity=0.9,
                    line=dict(color='#0097b8', width=0.5)),
        name='Quantum states',
        text=[f"State {i}" for i in range(len(states))],
        hovertemplate='<b>%{text}</b><br>x:%{x:.3f}  y:%{y:.3f}  z:%{z:.3f}<extra></extra>'
    )

    axis_len = 1.3
    for vec, col, label in [
        ([axis_len, 0, 0], '#ff4f6a', 'X'),
        ([0, axis_len, 0], '#39d07e', 'Y'),
        ([0, 0, axis_len], '#0097b8', 'Z'),
    ]:
        fig.add_scatter3d(
            x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
            mode='lines+text',
            line=dict(color=col, width=3),
            text=['', label],
            textfont=dict(color=col, size=12),
            textposition='top center',
            hoverinfo='skip', showlegend=False
        )

    bg = 'rgb(13, 17, 23)'
    grid = 'rgb(29, 42, 60)'

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', backgroundcolor=bg, gridcolor=grid, showbackground=True),
            yaxis=dict(title='Y', backgroundcolor=bg, gridcolor=grid, showbackground=True),
            zaxis=dict(title='Z', backgroundcolor=bg, gridcolor=grid, showbackground=True),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode='cube'
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        font=dict(family='JetBrains Mono, monospace', size=11, color='#c8d8e8'),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        hovermode='closest'
    )

    return fig