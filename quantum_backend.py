import numpy as np
import plotly.graph_objects as go
from functools import lru_cache

# =====================================================
# CONFIG
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
# PUERTAS CUÁNTICAS BÁSICAS
# =====================================================

def rx(theta):
    """Puerta RX(theta)"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex64)

def ry(theta):
    """Puerta RY(theta)"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex64)

def rz(theta):
    """Puerta RZ(theta)"""
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=np.complex64)

def cz_gate():
    """Puerta CZ"""
    return np.diag([1, 1, 1, -1]).astype(np.complex64)

# =====================================================
# OPERACIONES VECTORIZADAS
# =====================================================

def apply_single_qubit_gate(states, gate, target_qubit, num_qubits):
    """
    Aplica una puerta de 1 qubit a todos los estados en batch.
    
    Args:
        states: (batch_size, 2^num_qubits) - estados cuánticos
        gate: (2, 2) - matriz unitaria
        target_qubit: índice del qubit
        num_qubits: número total de qubits
    
    Returns:
        (batch_size, 2^num_qubits) - estados transformados
    """
    batch_size = states.shape[0]
    dim = 2**num_qubits
    
    # Reshape a (batch, 2, 2, ..., 2)
    s = states.reshape((batch_size,) + (2,)*num_qubits)
    
    # Mueve el qubit target al eje 1
    axes_perm = list(range(num_qubits + 1))
    axes_perm[1], axes_perm[target_qubit + 1] = axes_perm[target_qubit + 1], axes_perm[1]
    s = np.transpose(s, axes_perm)
    
    # Aplica gate
    s = s.reshape(batch_size, 2, -1)
    s = np.einsum('ij,bjk->bik', gate, s, dtype=np.complex64)
    
    # Revierte la transposición
    s = s.reshape((batch_size,) + (2,)*num_qubits)
    axes_inv = [0] * (num_qubits + 1)
    for i, a in enumerate(axes_perm):
        axes_inv[a] = i
    s = np.transpose(s, axes_inv)
    
    return s.reshape(batch_size, dim).astype(np.complex64)


def apply_two_qubit_gate(states, gate, q1, q2, num_qubits):
    """
    Aplica una puerta de 2 qubits a todos los estados en batch.
    
    Args:
        states: (batch_size, 2^num_qubits)
        gate: (4, 4) - matriz unitaria
        q1, q2: índices de los qubits
        num_qubits: número total de qubits
    
    Returns:
        (batch_size, 2^num_qubits)
    """
    batch_size = states.shape[0]
    dim = 2**num_qubits
    
    # Asegura q1 < q2
    if q1 > q2:
        q1, q2 = q2, q1
    
    s = states.reshape((batch_size,) + (2,)*num_qubits)
    
    # Mueve ambos qubits al frente
    axes_perm = list(range(num_qubits + 1))
    axes_perm[1], axes_perm[q1 + 1] = axes_perm[q1 + 1], axes_perm[1]
    if q2 > q1:
        axes_perm[2], axes_perm[q2 + 1] = axes_perm[q2 + 1], axes_perm[2]
    else:
        axes_perm[1], axes_perm[q2 + 1] = axes_perm[q2 + 1], axes_perm[1]
    
    s = np.transpose(s, axes_perm)
    
    # Aplica gate
    s = s.reshape(batch_size, 4, -1)
    s = np.einsum('ij,bjk->bik', gate, s, dtype=np.complex64)
    
    # Revierte transposición
    s = s.reshape((batch_size,) + (2,)*num_qubits)
    axes_inv = [0] * (num_qubits + 1)
    for i, a in enumerate(axes_perm):
        axes_inv[a] = i
    s = np.transpose(s, axes_inv)
    
    return s.reshape(batch_size, dim).astype(np.complex64)


# =====================================================
# ANCILLAS
# =====================================================

def add_ancillas(data_states, num_ancillas):
    """
    Añade ancillas en estado |0⟩ a los estados de datos.
    
    Args:
        data_states: (batch, 2^ndataq)
        num_ancillas: número de ancillas
    
    Returns:
        (batch, 2^(ndataq + num_ancillas))
    """
    batch_size = data_states.shape[0]
    ancilla_state = np.zeros(2**num_ancillas, dtype=np.complex64)
    ancilla_state[0] = 1.0
    
    # Producto de Kronecker fila a fila
    full = data_states[:, :, None] * ancilla_state[None, None, :]
    return full.reshape(batch_size, -1).astype(np.complex64)


def measure_and_collapse_ancillas(full_states, num_ancillas, num_dataq, seed=None):
    """
    Mide las ancillas y colapsa el estado a los qubits de datos.
    Cada estado obtiene un resultado de medición diferente (aleatorio).
    
    Args:
        full_states: (batch, 2^(ndataq + num_ancillas))
        num_ancillas: número de ancillas
        num_dataq: número de qubits de datos
        seed: seed para RNG
    
    Returns:
        (batch, 2^ndataq) - estados de datos post-medición
    """
    batch_size = full_states.shape[0]
    data_dim = 2**num_dataq
    ancilla_dim = 2**num_ancillas
    
    # Reshape a (batch, data_dim, ancilla_dim)
    s = full_states.reshape(batch_size, data_dim, ancilla_dim)
    
    # Probabilidades de cada resultado de medición
    probs = np.sum(np.abs(s)**2, axis=1)  # (batch, ancilla_dim)
    probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-10)
    
    # RNG
    rng = np.random.default_rng(seed)
    
    # Muestrea un resultado aleatorio por estado
    outcomes = np.array([
        rng.choice(ancilla_dim, p=probs[i])
        for i in range(batch_size)
    ])
    
    # Extrae estado de datos post-medición
    post_states = s[np.arange(batch_size), :, outcomes]  # (batch, data_dim)
    
    # Normaliza
    norms = np.linalg.norm(post_states, axis=1, keepdims=True)
    post_states = post_states / (norms + 1e-12)
    
    return post_states.astype(np.complex64)


# =====================================================
# CIRCUITO PARAMETRIZADO
# =====================================================

def apply_pqc_layer(states, params, num_qubits):
    """
    Aplica una capa del circuito parametrizado:
    RX(params[0]) - RY(params[1]) - ... - RX/RY en todos los qubits
    luego CZ entre qubits adyacentes.
    
    Args:
        states: (batch, 2^num_qubits)
        params: (2*num_qubits,) - ángulos para RX y RY
        num_qubits: número total de qubits
    
    Returns:
        (batch, 2^num_qubits)
    """
    # Puertas RX
    for q in range(num_qubits):
        gate = rx(float(params[2*q]))
        states = apply_single_qubit_gate(states, gate, q, num_qubits)
    
    # Puertas RY
    for q in range(num_qubits):
        gate = ry(float(params[2*q + 1]))
        states = apply_single_qubit_gate(states, gate, q, num_qubits)
    
    # Puertas CZ entre qubits adyacentes
    cz = cz_gate()
    for q in range(num_qubits - 1):
        states = apply_two_qubit_gate(states, cz, q, q+1, num_qubits)
    
    return states.astype(np.complex64)


def encode_conditioning(full_states, mu1, mu2, num_ancillas, num_dataq, num_qubits):
    """
    Codifica parámetros de condicionamiento (μ₁, μ₂) aplicando puertas a las ancillas.
    
    Args:
        full_states: (batch, 2^num_qubits)
        mu1, mu2: parámetros de condicionamiento
        num_ancillas: número de ancillas
        num_dataq: número de qubits de datos
        num_qubits: número total de qubits
    
    Returns:
        (batch, 2^num_qubits)
    """
    ry_gate = ry(float(mu1))
    rz_gate = rz(float(mu2))
    
    # Aplica a cada ancilla
    for i in range(num_ancillas):
        qubit_idx = num_dataq + i
        full_states = apply_single_qubit_gate(full_states, ry_gate, qubit_idx, num_qubits)
        full_states = apply_single_qubit_gate(full_states, rz_gate, qubit_idx, num_qubits)
    
    return full_states.astype(np.complex64)


# =====================================================
# CARGA DE PARÁMETROS
# =====================================================

@lru_cache(maxsize=3)
def load_params(distribution):
    """Carga parámetros entrenados desde archivo NPY."""
    if distribution not in PARAM_FILES:
        raise ValueError(f"Distribución no soportada: {distribution}")
    return np.load(PARAM_FILES[distribution], allow_pickle=False)


# =====================================================
# GENERACIÓN DE ESTADOS INICIALES
# =====================================================

def generate_haar_random_states(num_states, seed=42):
    """
    Genera estados cuánticos de 1 qubit uniformemente distribuidos
    en la esfera de Bloch (distribución Haar).
    
    Args:
        num_states: número de estados a generar
        seed: seed para reproducibilidad
    
    Returns:
        (num_states, 2) - estados complejos [α, β]
    """
    rng = np.random.default_rng(seed)
    
    # Distribución uniforme en la esfera de Bloch
    u = rng.uniform(0, 1, num_states)
    phi = rng.uniform(0, 2*np.pi, num_states)
    
    theta = np.arccos(1 - 2*u)
    alpha = np.cos(theta/2).astype(np.complex64)
    beta = (np.exp(1j*phi) * np.sin(theta/2)).astype(np.complex64)
    
    return np.stack([alpha, beta], axis=1)


# =====================================================
# SIMULACIÓN PRINCIPAL
# =====================================================

def run_simulation(distribution, mu1, mu2, nstates):
    """
    Ejecuta la simulación completa del modelo CQDD.
    
    Args:
        distribution: "Cluster States" | "Rings" | "Parallel Rings"
        mu1, mu2: parámetros de condicionamiento
        nstates: número de estados a simular
    
    Returns:
        (nstates, 2) - estados finales en la esfera de Bloch
    """
    # Configuración del modelo
    cfg = MODEL_CONFIG[distribution]
    num_ancillas = cfg["nancilla"]
    num_dataq = cfg["ndataq"]
    num_qubits = num_ancillas + num_dataq
    
    # Para Parallel Rings y Rotated Rings, μ₂ es siempre 0
    if distribution == "Parallel Rings" or distribution == "Rotated Rings":
        mu2 = 0.0
    
    # Carga parámetros entrenados
    params = load_params(distribution)
    
    if params.ndim != 3:
        raise ValueError(f"Parámetros deben tener shape (T, L, P), recibido {params.shape}")
    
    T, L, P = params.shape
    
    # Validación de dimensiones
    expected_P = 2 * num_qubits
    if P != expected_P:
        raise ValueError(
            f"Mismatch de dimensiones para '{distribution}': "
            f"esperado P={expected_P}, recibido {P}"
        )
    
    # Estados iniciales: distribuidos uniformemente en la esfera de Bloch
    states = generate_haar_random_states(nstates, seed=42)  # (nstates, 2)
    
    # Simulación: T pasos de denoising
    for t in range(T):
        # 1. Añade ancillas en estado |0⟩
        full_states = add_ancillas(states, num_ancillas)  # (nstates, 2^num_qubits)
        
        # 2. Codifica parámetros μ₁, μ₂ en las ancillas
        full_states = encode_conditioning(
            full_states, mu1, mu2, num_ancillas, num_dataq, num_qubits
        )
        
        # 3. Aplica L capas del circuito parametrizado
        for layer in range(L):
            full_states = apply_pqc_layer(full_states, params[t, layer, :], num_qubits)
        
        # 4. Mide ancillas y colapsa a estados de datos
        states = measure_and_collapse_ancillas(
            full_states, num_ancillas, num_dataq, seed=t
        )  # (nstates, 2)
    
    return states


# =====================================================
# COORDENADAS DE BLOCH
# =====================================================

def states_to_bloch_coords(states):
    """
    Convierte estados cuánticos |ψ⟩ = α|0⟩ + β|1⟩ a coordenadas de Bloch (x, y, z).
    
    Fórmulas:
        x = 2 Re(α* β)
        y = 2 Im(α* β)
        z = |α|² - |β|²
    
    Args:
        states: (num_states, 2) - estados [α, β]
    
    Returns:
        (xs, ys, zs) - coordenadas de Bloch
    """
    states = np.asarray(states, dtype=np.complex64)
    
    alpha = states[:, 0]
    beta = states[:, 1]
    ab_conj = np.conj(alpha) * beta
    
    xs = 2.0 * np.real(ab_conj)
    ys = 2.0 * np.imag(ab_conj)
    zs = np.abs(alpha)**2 - np.abs(beta)**2
    
    return (
        xs.astype(float),
        ys.astype(float),
        zs.astype(float)
    )


# =====================================================
# VISUALIZACIÓN PLOTLY
# =====================================================

def create_bloch_plotly(states):
    """
    Crea una visualización interactiva de la esfera de Bloch con Plotly.
    
    Args:
        states: (num_states, 2) - estados cuánticos
    
    Returns:
        plotly.graph_objects.Figure
    """
    xs, ys, zs = states_to_bloch_coords(states)
    
    # Genera esfera de Bloch
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 30)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))
    
    fig = go.Figure()
    
    # Superficie de la esfera
    fig.add_surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        opacity=0.12,
        colorscale='Greys',
        showscale=False,
        hoverinfo='skip',
        name='Bloch sphere'
    )
    
    # Puntos de estados
    fig.add_scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(
            size=4,
            color='#00d4ff',
            opacity=0.85,
            line=dict(color='#0097b8', width=0.5)
        ),
        name='Quantum states',
        text=[f"State {i}" for i in range(len(states))],
        hovertemplate='<b>%{text}</b><br>x: %{x:.3f}  y: %{y:.3f}  z: %{z:.3f}<extra></extra>'
    )
    
    # Ejes cartesianos
    axis_length = 1.3
    axes_config = [
        ([axis_length, 0, 0], '#ff4f6a', 'X'),
        ([0, axis_length, 0], '#39d07e', 'Y'),
        ([0, 0, axis_length], '#0097b8', 'Z'),
    ]
    
    for vector, color, label in axes_config:
        fig.add_scatter3d(
            x=[0, vector[0]],
            y=[0, vector[1]],
            z=[0, vector[2]],
            mode='lines+text',
            line=dict(color=color, width=3),
            text=['', label],
            textfont=dict(color=color, size=12),
            textposition='top center',
            hoverinfo='skip',
            showlegend=False
        )
    
    # Configuración layout
    bg_color = 'rgb(13, 17, 23)'
    grid_color = 'rgb(29, 42, 60)'
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X',
                backgroundcolor=bg_color,
                gridcolor=grid_color,
                showbackground=True,
                zerolinecolor=grid_color
            ),
            yaxis=dict(
                title='Y',
                backgroundcolor=bg_color,
                gridcolor=grid_color,
                showbackground=True,
                zerolinecolor=grid_color
            ),
            zaxis=dict(
                title='Z',
                backgroundcolor=bg_color,
                gridcolor=grid_color,
                showbackground=True,
                zerolinecolor=grid_color
            ),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode='cube'
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        font=dict(family='JetBrains Mono, monospace', size=11, color='#c8d8e8'),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=True,
        hovermode='closest'
    )
    
    return fig