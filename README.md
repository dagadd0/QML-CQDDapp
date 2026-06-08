# CQDD App — Interactive Quantum State Generator

An interactive web application for generating and visualizing quantum state distributions using the [Conditional Quantum Diffusion Denoising (CQDD) model](https://github.com/dagadd0/QML-CQDDmodel).

🌐 **Live Demo**: [Deploy on Streamlit Cloud](https://cqddapp.streamlit.app/)  
🚀 **Tech Stack**: Streamlit + JAX + TensorCircuit + Plotly

## Overview

This application provides an intuitive interface to the CQDD quantum machine learning model. Generate quantum states on the Bloch sphere by specifying conditioning parameters, visualize them in real-time, and explore different target distributions—all without writing code.

## Features

✨ **Interactive Conditioning**
- Slider or manual input for conditioning parameters (μ₁, μ₂)
- Support for Cartesian, polar, and fractional angle formats (e.g., `π/2`, `2π`, `pi/4`)
- Real-time parameter validation and formatting

🎨 **Real-time Visualization**
- Interactive 3D Bloch sphere visualization with Plotly
- Configurable state count (10-1000 states)

📊 **Multiple Distributions**
- **Cluster States**: Polar point distributions on the Bloch sphere
- **Rotated Rings**: XY rotated ring structures
- **Parallel Rings**: Parallel ring configurations



## Installation

### Requirements

- Python 3.8+
- Streamlit
- TensorCircuit
- JAX
- Plotly
- NumPy



The app will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Select Distribution**: Choose from Cluster States, Rings, or Parallel Rings
2. **Adjust Parameters**:
   - Set number of states (10-1000)
   - Configure μ₁ and μ₂ via slider or manual input
3. **Run Simulation**: Click "Run simulation" button
4. **Explore Results**:
   - View 3D Bloch sphere visualization
   - Check circuit diagram and configuration details
   - Download trained parameters if needed



## Project Structure

```
CQDD-app/
├── app.py                          # Main Streamlit application
├── quantum_backend.py              # Backend simulation & visualization
├── requirements.txt                # Python dependencies
├── circuit.png                     # PQC circuit diagram
│
├── params_clusters.npy             # Trained parameters: Cluster States
├── params_rings.npy                # Trained parameters: Rings
├── params_ringsparalel.npy         # Trained parameters: Parallel Rings
│
├── Other trained params/           # Alternative parameter sets
│   ├── params.npy
│   └── params_ringsparalel.npy
│
├── __pycache__/                    # Python cache (auto-generated)
└── README.md
```

## Configuration

### app.py

Key constants that can be customized:

```python
# Load parameter files
PARAM_FILES = {
    "Cluster States": "params_clusters.npy",
    "Rings": "params_rings.npy",
    "Parallel Rings": "params_ringsparalel.npy",
}

# Model configuration (qubits per distribution)
MODEL_CONFIG = {
    "Cluster States": {"nancilla": 2, "ndataq": 1},
    "Rings": {"nancilla": 3, "ndataq": 1},
    "Parallel Rings": {"nancilla": 3, "ndataq": 1},
}
```


## Model Architecture

### Parameter Shapes

| Distribution | Qubits | Params Shape | Description |
|---|---|---|---|
| Cluster States | 3 (1 data + 2 ancilla) | (T, L, 6) | 6 polar points |
| Rings | 4 (1 data + 3 ancilla) | (T, L, 8) | Parallel rings |
| Parallel Rings | 4 (1 data + 3 ancilla) | (T, L, 8) | Tunable rings |

where:
- **T** = Denoising steps (typically 20)
- **L** = PQC layers per step (typically 12)
- **6, 8** = 2 × num_qubits (rotation angles per qubit)

### Circuit Design

The parameterized quantum circuit (PQC) follows this structure at each denoising step:

```
Input states (Haar-random or partially denoised)
        ↓
   Encode μ into ancillas (RY, RZ rotations)
        ↓
   Variational block (L layers):
      For each layer:
        - RX rotation on all qubits
        - RY rotation on all qubits
        - CZ entanglement gates
        ↓
   Measure and discard ancillas
        ↓
   Output states (evolved toward target)
```

## Customization

### Adding a New Distribution

1. Train a model for your distribution using the main CQDD training code
2. Save parameters as `params_mynewdist.npy`
3. Add to `PARAM_FILES`:
   ```python
   PARAM_FILES["My Distribution"] = "params_mynewdist.npy"
   ```
4. Add to `MODEL_CONFIG`:
   ```python
   MODEL_CONFIG["My Distribution"] = {"nancilla": N_ANCILLA, "ndataq": N_DATA}
   ```
5. Restart the app



## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Additional distribution types (e.g., superposition states, entangled pairs)
- [ ] Parameter sweep/animation features
- [ ] Export state data as CSV/JSON
- [ ] Comparison mode (two distributions side-by-side)
- [ ] Animation of denoising process over timesteps



## Related Projects

- [CQDD Training Code](https://github.com/dagadd0/QML-CQDDmodel/tree/main) - Full training pipeline
- [TensorCircuit](https://github.com/tencent/tensorcircuit) - Quantum circuit framework
- [JAX](https://github.com/google/jax) - Automatic differentiation and compilation


## Changelog

### v1.0.0 (June 2024)
- Initial release
- Support for 3 distribution types
- Interactive Bloch sphere visualization
- Dark scientific theme
- Flexible angle input formats

---

**Status**: Active Development  
**Last Updated**: June 2026  