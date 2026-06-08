import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import expm
import plotly.graph_objects as go

# =====================================================
# CONFIGURACIÓN GLOBAL
# =====================================================

PARAM_FILES = {
    "Cluster States": "params_clusters.npy",
    "Rings": "params_rings.npy",
    "Parallel Rings": "params_ringsparalel.npy",
}

MODEL_CONFIG = {
    "Cluster States": {"nancilla": 2, "ndataq": 1},
    "Rings": {"nancilla": 3, "ndataq": 1},
    "Parallel Rings": {"nancilla": 3, "ndataq": 1},
}

# =====================================================
# MATRICES DE PAULI Y GATES CUÁNTICOS
# =====================================================

# Matrices de Pauli
I = np.array([[1, 0], [0, 1]], dtype=np.complex64)
X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)

# Gates unitarios
def RX(theta):
    """Rotación alrededor del eje X"""
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=np.complex64)

def RY(theta):
    """Rotación alrededor del eje Y"""
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=np.complex64)

def RZ(theta):
    """Rotación alrededor del eje Z"""
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=np.complex64)

def CZ_gate():
    """Compuerta CZ (Control-Z)"""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=np.complex64)

# =====================================================
# CARGA DE PARÁMETROS
# =====================================================

def load_params(distribution):
    """Carga los parámetros entrenados del archivo .npy"""
    if distribution not in PARAM_FILES:
        raise ValueError(f"Distribución no soportada: {distribution}")
    return np.load(PARAM_FILES[distribution], allow_pickle=False)


# =====================================================
# ESTADOS ALEATORIOS EN BLOCH
# =====================================================

def random_bloch_states(key, N):
    """
    Genera N estados cuánticos aleatorios uniformemente distribuidos
    en la esfera de Bloch.
    
    Args:
        key: JAX random key
        N: número de estados
    
    Returns:
        Array de shape (N, 2) con amplitudes [α, β]
    """
    key_theta, key_phi = jax.random.split(key)

    # Distribución uniforme en la esfera
    u = jax.random.uniform(key_theta, (N,))
    theta = jnp.arccos(1 - 2 * u)

    phi = 2 * jnp.pi * jax.random.uniform(key_phi, (N,))

    alpha = jnp.cos(theta / 2)
    beta = jnp.exp(1j * phi) * jnp.sin(theta / 2)

    return jnp.stack([alpha, beta], axis=1)


# =====================================================
# OPERACIONES CON ESPACIOS TENSORIALES
# =====================================================

def kron_product(*matrices):
    """
    Calcula el producto de Kronecker de múltiples matrices.
    Equivalente a np.kron pero para múltiples matrices.
    """
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result

def apply_single_qubit_gate(state, gate, qubit, nqubits):
    """
    Aplica una compuerta de un qubit a un estado.
    
    Args:
        state: vector de estado de shape (2**nqubits,)
        gate: matriz unitaria 2x2
        qubit: índice del qubit (0 a nqubits-1)
        nqubits: número total de qubits
    
    Returns:
        estado transformado
    """
    # Construye la matriz total I ⊗ ... ⊗ gate ⊗ ... ⊗ I
    matrices = []
    for i in range(nqubits):
        if i == qubit:
            matrices.append(gate)
        else:
            matrices.append(I)
    
    total_matrix = kron_product(*matrices)
    return total_matrix @ state

def apply_two_qubit_gate(state, gate, qubit1, qubit2, nqubits):
    """
    Aplica una compuerta de dos qubits (como CZ).
    
    Args:
        state: vector de estado
        gate: matriz unitaria 4x4
        qubit1, qubit2: índices de qubits
        nqubits: número total de qubits
    
    Returns:
        estado transformado
    """
    # Asegura que qubit1 < qubit2
    if qubit1 > qubit2:
        qubit1, qubit2 = qubit2, qubit1
    
    # Reordena qubits para aplicar la compuerta en los primeros 2 qubits
    # (simplificado: asume qubits adyacentes)
    matrices = []
    for i in range(nqubits):
        if i < qubit1 or i > qubit2:
            matrices.append(I)
        elif i == qubit1:
            continue  # Manejado por la compuerta de 2 qubits
        elif i == qubit2:
            continue  # Manejado por la compuerta de 2 qubits
    
    # Versión simplificada: construye matriz de permutación
    # Para mayor precisión, usar librería más avanzada
    total_matrix = kron_product(*([I] * qubit1 + [gate] + [I] * (nqubits - qubit2 - 1)))
    return total_matrix @ state


# =====================================================
# MEDICIÓN DE ANCILLAS
# =====================================================

def measure_ancillas_classical(state, nancilla, ndataq):
    """
    Realiza medición proyectiva en las ancillas y retorna estado colapsado.
    
    Args:
        state: vector de estado completo de shape (2**nqubits,)
        nancilla: número de ancillas
        ndataq: número de qubits de datos
    
    Returns:
        estado de datos post-medición de shape (2**ndataq,)
    """
    nqubits = nancilla + ndataq
    
    # Reshape: datos × ancillas
    reshaped = state.reshape((2**ndataq, 2**nancilla))
    
    # Calcula probabilidades de medir cada resultado de ancilla
    probs = np.sum(np.abs(reshaped) ** 2, axis=0)
    probs = probs / np.sum(probs)  # Normaliza
    
    # Elige resultado de medición (determinístico en simulación: máx probabilidad)
    m_result = np.argmax(probs)
    
    # Obtiene estado de datos post-colapso
    post_state = reshaped[:, m_result]
    post_state = post_state / (np.linalg.norm(post_state) + 1e-12)
    
    return post_state


# =====================================================
# CIRCUITO PARAMETRIZADO
# =====================================================

def apply_pqc_layer(state, params, nqubits):
    """
    Aplica una capa del circuito parametrizado.
    
    Estructura:
      1. RX en todos los qubits
      2. RY en todos los qubits
      3. CZ entre qubits adyacentes
    
    Args:
        state: vector de estado
        params: array de shape (2*nqubits,) con ángulos [rx_0, ry_0, rx_1, ry_1, ...]
        nqubits: número total de qubits
    
    Returns:
        estado transformado
    """
    # Extrae ángulos
    rx_angles = params[0::2]  # Índices pares
    ry_angles = params[1::2]  # Índices impares
    
    # Aplica RX en todos los qubits
    for i in range(nqubits):
        state = apply_single_qubit_gate(state, RX(rx_angles[i]), i, nqubits)
    
    # Aplica RY en todos los qubits
    for i in range(nqubits):
        state = apply_single_qubit_gate(state, RY(ry_angles[i]), i, nqubits)
    
    # Aplica CZ entre qubits adyacentes
    for i in range(nqubits - 1):
        state = apply_two_qubit_gate(state, CZ_gate(), i, i+1, nqubits)
    
    return state


def transform_state(state, params, mu, nancilla, ndataq, nlayers):
    """
    Transforma un estado usando el circuito parametrizado con ancillas.
    
    Proceso:
      1. Prepara ancillas (todas en |0⟩)
      2. Encoding: RY(μ₁) y RZ(μ₂) en ancillas
      3. Aplica nlayers capas variacionales
      4. Mide ancillas y retorna estado de datos
    
    Args:
        state: estado inicial de datos de shape (2**ndataq,)
        params: parámetros de PQC de shape (nlayers, 2*nqubits)
        mu: conditioning parameters [μ₁, μ₂]
        nancilla: número de ancillas
        ndataq: número de qubits de datos
        nlayers: número de capas del PQC
    
    Returns:
        estado de datos transformado de shape (2**ndataq,)
    """
    nqubits = nancilla + ndataq
    
    # Prepara estado con ancillas: |ψ⟩ ⊗ |0⟩^⊗nancilla
    ancilla = np.zeros(2**nancilla, dtype=np.complex64)
    ancilla[0] = 1.0
    full_state = np.kron(state, ancilla)
    
    # Encoding en ancillas
    mu1, mu2 = mu[0], mu[1]
    for i in range(nancilla):
        # RY(μ₁) en ancilla i
        full_state = apply_single_qubit_gate(
            full_state, 
            RY(mu1), 
            ndataq + i, 
            nqubits
        )
        # RZ(μ₂) en ancilla i
        full_state = apply_single_qubit_gate(
            full_state,
            RZ(mu2),
            ndataq + i,
            nqubits
        )
    
    # Aplica capas variacionales
    for layer in range(nlayers):
        full_state = apply_pqc_layer(full_state, params[layer], nqubits)
    
    # Mide ancillas y retorna estado de datos
    data_state = measure_ancillas_classical(full_state, nancilla, ndataq)
    
    return data_state


# =====================================================
# BATCHED TRANSFORMATION
# =====================================================

def batched_transform(states, params, mu, nancilla, ndataq, nlayers):
    """
    Aplica transform_state a un batch de estados.
    
    Args:
        states: array de shape (nbatch, 2**ndataq)
        params: parámetros compartidos de shape (nlayers, 2*nqubits)
        mu: conditioning [μ₁, μ₂]
        nancilla, ndataq, nlayers: configuración
    
    Returns:
        array de estados transformados de shape (nbatch, 2**ndataq)
    """
    transformed = []
    for state in states:
        t_state = transform_state(state, params, mu, nancilla, ndataq, nlayers)
        transformed.append(t_state)
    return np.array(transformed)


# =====================================================
# SIMULACIÓN PRINCIPAL
# =====================================================

def run_simulation(distribution, mu1, mu2, nstates):
    """
    Ejecuta la simulación completa del modelo CQDD.
    
    Args:
        distribution: "Cluster States", "Rings", o "Parallel Rings"
        mu1, mu2: parámetros de condicionamiento
        nstates: número de estados a generar
    
    Returns:
        array de estados finales de shape (nstates, 2) con amplitudes [α, β]
    """
    
    if distribution not in MODEL_CONFIG:
        raise ValueError(f"Distribución no soportada: {distribution}")
    
    cfg = MODEL_CONFIG[distribution]
    nancilla = cfg["nancilla"]
    ndataq = cfg["ndataq"]
    nqubits = nancilla + ndataq
    
    # Para anillos paralelos, μ₂ no se usa
    if distribution == "Parallel Rings":
        mu2 = 0.0
    
    # Carga parámetros entrenados
    params = load_params(distribution)
    
    if params.ndim != 3:
        raise ValueError(
            f"El tensor de parámetros debe tener shape (T, L, P). "
            f"Recibido: {params.shape}"
        )
    
    T_steps, L_layers, P_params = params.shape
    
    # Valida dimensión de parámetros
    expected_P = 2 * nqubits
    if P_params != expected_P:
        raise ValueError(
            f"Mismatch para '{distribution}': "
            f"se esperaba último eje {expected_P} (= 2 * {nqubits}), "
            f"pero llegó {P_params}. Shape completa: {params.shape}"
        )
    
    # Inicializa estados aleatorios en la esfera de Bloch
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    
    # Estados iniciales en 1 qubit (2**ndataq = 2)
    initial_states_1qubit = random_bloch_states(subkey, nstates)
    
    # Itera sobre pasos de denoising
    current_states = np.array(initial_states_1qubit)
    
    for t in range(T_steps):
        # Convierte amplitudes a vectores de estado para ndataq qubits
        states_vectors = []
        for state_2d in current_states:
            # Para ndataq=1: amplitudes [α, β] es suficiente
            states_vectors.append(state_2d.astype(np.complex64))
        states_vectors = np.array(states_vectors)
        
        # Aplica transformación
        mu = [mu1, mu2]
        transformed_vectors = batched_transform(
            states_vectors,
            params[t],
            mu,
            nancilla,
            ndataq,
            L_layers
        )
        
        current_states = transformed_vectors
    
    return current_states


# =====================================================
# CONVERSIÓN A COORDENADAS DE BLOCH
# =====================================================

def states_to_bloch_coords(states):
    """
    Convierte estados cuánticos a coordenadas en la esfera de Bloch.
    
    Mapeo para 1 qubit:
        |ψ⟩ = α|0⟩ + β|1⟩ → (x, y, z)
    
    donde:
        x = 2 Re(α* β)
        y = 2 Im(α* β)
        z = |α|² - |β|²
    
    Args:
        states: array de shape (nstates, 2) con [α, β]
    
    Returns:
        tuple (xs, ys, zs) - arrays de coordenadas
    """
    xs, ys, zs = [], [], []
    
    for state in states:
        alpha = complex(state[0])
        beta = complex(state[1])
        
        conj_alpha_beta = np.conj(alpha) * beta
        
        x = 2 * np.real(conj_alpha_beta)
        y = 2 * np.imag(conj_alpha_beta)
        z = np.abs(alpha)**2 - np.abs(beta)**2
        
        xs.append(x)
        ys.append(y)
        zs.append(z)
    
    return np.array(xs), np.array(ys), np.array(zs)


# =====================================================
# VISUALIZACIÓN CON PLOTLY
# =====================================================

def create_bloch_plotly(states):
    """
    Crea visualización interactiva de la esfera de Bloch con Plotly.
    
    Args:
        states: array de shape (nstates, 2) con amplitudes
    
    Returns:
        Figura de Plotly
    """
    
    xs, ys, zs = states_to_bloch_coords(states)
    
    # Genera esfera de Bloch como referencia
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 30)
    
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))
    
    fig = go.Figure()
    
    # Añade esfera de referencia
    fig.add_surface(
        x=sphere_x,
        y=sphere_y,
        z=sphere_z,
        opacity=0.15,
        colorscale='Greys',
        showscale=False,
        name='Bloch sphere',
        hoverinfo='skip'
    )
    
    # Añade puntos de estados
    fig.add_scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers',
        marker=dict(
            size=4,
            color='#00d4ff',
            opacity=0.85,
            line=dict(color='#0097b8', width=0.5)
        ),
        name='Quantum states',
        text=[f"State {i}" for i in range(len(states))],
        hovertemplate='<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
    )
    
    # Añade ejes de referencia
    axis_length = 1.3
    
    # Eje X (rojo)
    fig.add_scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='#ff4f6a', width=3),
        name='X axis',
        hoverinfo='skip',
        showlegend=False
    )
    
    # Eje Y (verde)
    fig.add_scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines',
        line=dict(color='#39d07e', width=3),
        name='Y axis',
        hoverinfo='skip',
        showlegend=False
    )
    
    # Eje Z (azul)
    fig.add_scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines',
        line=dict(color='#0097b8', width=3),
        name='Z axis',
        hoverinfo='skip',
        showlegend=False
    )
    
    # Configura el layout
    fig.update_layout(
        title='Bloch Sphere Visualization',
        scene=dict(
            xaxis=dict(
                title='X',
                backgroundcolor='rgb(13, 17, 23)',
                gridcolor='rgb(29, 42, 60)',
                showbackground=True,
                zerolinecolor='rgb(100, 100, 100)'
            ),
            yaxis=dict(
                title='Y',
                backgroundcolor='rgb(13, 17, 23)',
                gridcolor='rgb(29, 42, 60)',
                showbackground=True,
                zerolinecolor='rgb(100, 100, 100)'
            ),
            zaxis=dict(
                title='Z',
                backgroundcolor='rgb(13, 17, 23)',
                gridcolor='rgb(29, 42, 60)',
                showbackground=True,
                zerolinecolor='rgb(100, 100, 100)'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            aspectmode='cube'
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        font=dict(
            family='JetBrains Mono, monospace',
            size=11,
            color='#c8d8e8'
        ),
        paper_bgcolor='rgba(13, 17, 23, 0)',
        plot_bgcolor='rgba(13, 17, 23, 0)',
        showlegend=True,
        hovermode='closest'
    )
    
    return fig