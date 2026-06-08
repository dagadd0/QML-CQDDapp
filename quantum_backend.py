import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc
import plotly.graph_objects as go

# =====================================================
# CONFIGURACIÓN GLOBAL
# =====================================================

tc.set_dtype("complex64")
tc.set_backend("jax")

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
# CARGA DE PARÁMETROS
# =====================================================

def load_params(distribution):
    if distribution not in PARAM_FILES:
        raise ValueError(f"Distribución no soportada: {distribution}")
    return jnp.array(np.load(PARAM_FILES[distribution], allow_pickle=False))


# =====================================================
# ESTADOS ALEATORIOS BLOCH
# =====================================================

def random_bloch_states(key, N):
    key_theta, key_phi = jax.random.split(key)

    u = jax.random.uniform(key_theta, (N,))
    theta = jnp.arccos(1 - 2 * u)

    phi = 2 * jnp.pi * jax.random.uniform(key_phi, (N,))

    alpha = jnp.cos(theta / 2)
    beta = jnp.exp(1j * phi) * jnp.sin(theta / 2)

    return jnp.stack([alpha, beta], axis=1)


# =====================================================
# ANCILLAS
# =====================================================

def add_ancilla(state, nancilla):
    ancilla = (
        jnp.zeros((2**nancilla,), dtype=state.dtype)
        .at[0].set(1)
    )
    return jnp.kron(state, ancilla)


batched_addancilla = jax.vmap(
    add_ancilla,
    in_axes=(0, None)
)


# =====================================================
# MEDICIÓN DE ANCILLAS
# =====================================================

def measure_ancillas_single(key, state, na, n):
    reshaped = jnp.reshape(state, (2**n, 2**na))
    probs = jnp.sum(jnp.abs(reshaped) ** 2, axis=0)

    key, subkey = jax.random.split(key)
    m_res = jax.random.categorical(subkey, jnp.log(probs + 1e-12))

    post_state = reshaped[:, m_res]
    norm = jnp.linalg.norm(post_state)

    return post_state / (norm + 1e-12)


# =====================================================
# FACTORÍA DEL CIRCUITO
# =====================================================

def make_batched_transform(nqubits, nancilla, ndataq, nlayers):
    """
    Devuelve una función vectorizada/jit con nqubits, nancilla y nlayers
    como constantes Python cerradas, evitando TracerBoolConversionError.
    """

    def transform_singlestate(params, mu, current_state, key):
        c = tc.Circuit(
            nqubits,
            inputs=current_state
        )

        # encoding en ancillas
        for i in range(nancilla):
            c.ry(ndataq + i, theta=mu[0])
            c.rz(ndataq + i, theta=mu[1])

        # capas variacionales
        for l in range(nlayers):
            for i in range(nqubits):
                c.rx(i, theta=params[l][2 * i])
                c.ry(i, theta=params[l][2 * i + 1])

            for i in range(nqubits - 1):
                c.cz(i, i + 1)

        full_state = c.state()

        data_state = measure_ancillas_single(
            key,
            full_state,
            nancilla,
            ndataq
        )

        return data_state

    return jax.jit(
        jax.vmap(
            transform_singlestate,
            in_axes=(
                None,   # params
                None,   # mu
                0,      # batch states
                0       # keys
            )
        )
    )


# =====================================================
# SIMULACIÓN PRINCIPAL
# =====================================================

def run_simulation(distribution, mu1, mu2, nstates):
    if distribution not in MODEL_CONFIG:
        raise ValueError(f"Distribución no soportada: {distribution}")

    cfg = MODEL_CONFIG[distribution]
    nancilla = cfg["nancilla"]
    ndataq = cfg["ndataq"]
    nqubits = nancilla + ndataq

    if distribution == "Parallel Rings":
        mu2 = 0.0

    key = jax.random.PRNGKey(0)

    params = load_params(distribution)
    if params.ndim != 3:
        raise ValueError(
            f"El tensor de parámetros debe tener shape (T, L, P). "
            f"Recibido: {params.shape}"
        )

    T_local, L_local, P_local = params.shape

    expected_P = 2 * nqubits
    if P_local != expected_P:
        raise ValueError(
            f"Mismatch para '{distribution}': "
            f"se esperaba último eje {expected_P} (= 2 * {nqubits}), "
            f"pero llegó {P_local}. Shape completa: {params.shape}"
        )

    transform = make_batched_transform(
        nqubits=nqubits,
        nancilla=nancilla,
        ndataq=ndataq,
        nlayers=L_local
    )

    key, subkey = jax.random.split(key)
    states = random_bloch_states(subkey, nstates)
    states = [states]

    mu = [mu1, mu2]

    for t in range(T_local):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, nstates)

        transformed = transform(
            params[t],
            mu,
            batched_addancilla(states[-1], nancilla),
            keys
        )

        states.append(transformed)

    return states[-1]


# =====================================================
# BLOCH COORDS
# =====================================================

def states_to_bloch_coords(states):
    xs, ys, zs = [], [], []

    for vec in states:
        alpha = complex(vec[0])
        beta = complex(vec[1])

        x = 2 * np.real(np.conj(alpha) * beta)
        y = 2 * np.imag(np.conj(alpha) * beta)
        z = np.abs(alpha) ** 2 - np.abs(beta) ** 2

        xs.append(x)
        ys.append(y)
        zs.append(z)

    return np.array(xs), np.array(ys), np.array(zs)


# =====================================================
# PLOTLY BLOCH SPHERE
# =====================================================

def create_bloch_plotly(states):
    xs, ys, zs = states_to_bloch_coords(states)

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()

    fig.add_surface(
        x=sphere_x,
        y=sphere_y,
        z=sphere_z,
        opacity=0.15,
        showscale=False
    )

    fig.add_scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(size=3, color="red")
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube"
        ),
        height=700,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig