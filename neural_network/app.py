import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
import io
import tempfile
import os

# Helper function to convert matplotlib figure to PIL Image
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return Image.open(buf)

st.set_page_config(page_title="Neural Network Deep Dive", layout="wide")
st.title("🧠 Neural Network: Complete Visual Workflow")

st.markdown("""
Explore how neural networks learn through **forward propagation**, **loss calculation**, 
**backpropagation**, and **gradient descent optimization**.
""")

# Sidebar navigation
mode = st.sidebar.radio(
    "**Select Topic:**",
    [
        "�️ Network Architecture",
        "�🔄 Forward Propagation",
        "⚡ Activation Functions",
        "📉 Loss Calculation",
        "🔙 Backpropagation",
        "🎯 Gradient Descent (SGD, Batch, Mini-Batch)",
    ],
    index=0
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0: NETWORK ARCHITECTURE - DATA FLOW VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

if mode == "🕸️ Network Architecture":
    st.header("🕸️ Neural Network Architecture: Data Flow Visualization")
    
    st.markdown("""
    See how input data flows through a neural network, getting transformed at each layer.
    Darker colors represent higher activation values.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        num_layers = st.slider("Number of Layers", 2, 6, 3)
    with col2:
        input_nodes = st.slider("Input Features", 2, 8, 4)
    with col3:
        hidden_nodes = st.slider("Hidden Units per Layer", 2, 8, 4)
    with col4:
        visualization_type = st.selectbox("View Type", ["Static Network", "Data Flow Animation"])
    
    # Create network data
    np.random.seed(42)
    
    # Generate layer sizes
    layer_sizes = [input_nodes] + [hidden_nodes] * (num_layers - 1)
    
    # Generate random activations for each layer
    activations = []
    current_activation = np.random.rand(input_nodes)
    activations.append(current_activation)
    
    for i in range(1, len(layer_sizes)):
        # Simulate forward pass with some transformation
        w = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * 0.5
        b = np.random.randn(layer_sizes[i]) * 0.1
        z = np.dot(current_activation, w) + b
        current_activation = np.maximum(0, z)  # ReLU
        current_activation = current_activation / (np.max(current_activation) + 1e-8)  # Normalize
        activations.append(current_activation)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Layout parameters
    layer_spacing = 3
    max_nodes = max(layer_sizes)
    neuron_size = 150
    
    # Draw network
    neuron_positions = {}
    
    for layer_idx, num_nodes in enumerate(layer_sizes):
        x_pos = layer_idx * layer_spacing
        y_positions = np.linspace(-(num_nodes - 1) / 2, (num_nodes - 1) / 2, num_nodes) * 1.5
        neuron_positions[layer_idx] = [(x_pos, y) for y in y_positions]
        
        # Get activations for this layer
        layer_activations = activations[layer_idx]
        
        # Draw neurons
        for node_idx, (x, y) in enumerate(neuron_positions[layer_idx]):
            # Color based on activation value
            activation_value = layer_activations[node_idx] if node_idx < len(layer_activations) else 0
            color_intensity = activation_value
            
            # Create color (blue to red gradient based on activation)
            color = plt.cm.RdYlBu_r(color_intensity)
            
            circle = plt.Circle((x, y), 0.25, color=color, ec="black", linewidth=2.5, zorder=5, alpha=0.8)
            ax.add_patch(circle)
            
            # Add activation value inside neuron
            ax.text(x, y, f"{activation_value:.2f}", ha="center", va="center", 
                   fontsize=8, fontweight="bold", color="white" if color_intensity > 0.5 else "black", zorder=6)
    
    # Draw connections
    for layer_idx in range(len(layer_sizes) - 1):
        from_positions = neuron_positions[layer_idx]
        to_positions = neuron_positions[layer_idx + 1]
        
        # Generate random weights for visualization
        np.random.seed(layer_idx)
        weights = np.random.randn(len(from_positions), len(to_positions))
        weights = np.abs(weights)  # Make all weights positive for visualization
        weights = weights / weights.max()  # Normalize to 0-1
        
        for from_idx, (x1, y1) in enumerate(from_positions):
            for to_idx, (x2, y2) in enumerate(to_positions):
                weight_strength = weights[from_idx, to_idx]
                
                # Draw line with thickness and color based on weight strength
                ax.plot([x1, x2], [y1, y2], 
                       color=plt.cm.viridis(weight_strength),
                       linewidth=weight_strength * 3 + 0.5,
                       alpha=0.4,
                       zorder=1)
    
    # Add layer labels
    layer_names = ["INPUT"] + [f"HIDDEN {i+1}" for i in range(num_layers - 2)] + (["OUTPUT"] if num_layers > 1 else [])
    for layer_idx, name in enumerate(layer_names):
        x_pos = layer_idx * layer_spacing
        ax.text(x_pos, max([y for _, y in neuron_positions[layer_idx]]) + 1.2, 
               name, fontsize=12, fontweight="bold", ha="center",
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Neuron Activation', rotation=270, labelpad=20, fontsize=11)
    
    ax.set_xlim(-1, (num_layers - 1) * layer_spacing + 1)
    ax.set_ylim(-(max_nodes) * 1, (max_nodes) * 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Neural Network Architecture - Input Data Flow Through Layers", 
                fontsize=14, fontweight="bold", pad=20)
    
    plt.tight_layout()
    st.image(fig_to_pil(fig), width="stretch")
    plt.close(fig)
    
    # Information panel
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Network Statistics")
        total_params = sum((layer_sizes[i] * layer_sizes[i+1]) for i in range(len(layer_sizes)-1))
        st.write(f"**Layers:** {num_layers}")
        st.write(f"**Total Parameters:** {total_params:,}")
        st.write(f"**Input Size:** {input_nodes}")
    
    with col2:
        st.markdown("### 🔍 Layer Details")
        for i, size in enumerate(layer_sizes):
            st.write(f"Layer {i}: {size} neurons")
    
    with col3:
        st.markdown("### 🎨 Color Guide")
        st.write("🔵 **Blue**: Low activation (near 0)")
        st.write("🟡 **Yellow**: Medium activation (0.5)")
        st.write("🔴 **Red**: High activation (near 1)")
    
    # Add animation option
    if visualization_type == "Data Flow Animation":
        st.markdown("---")
        st.markdown("### 🎬 Animated Data Flow")
        
        steps = st.slider("Animation Speed", 1, 10, 5, help="Higher = faster")
        
        # Create animated version
        fig_anim = plt.figure(figsize=(16, 10))
        ax_anim = fig_anim.add_subplot(111)
        
        frames_data = []
        for frame in range(num_layers * 3):
            frame_data = []
            for layer_idx in range(num_layers):
                # Simulate wave passing through network
                if frame >= layer_idx * 3:
                    intensity = min(1.0, (frame - layer_idx * 3) / 3)
                    noise = np.random.rand(layer_sizes[layer_idx]) * 0.3
                    layer_act = np.minimum(1.0, activations[layer_idx] * intensity + noise)
                else:
                    layer_act = np.zeros(layer_sizes[layer_idx])
                frame_data.append(layer_act)
            frames_data.append(frame_data)
        
        # Create animation
        def animate_frame(frame_idx):
            ax_anim.clear()
            
            frame_activations = frames_data[frame_idx]
            
            # Draw network for this frame
            for layer_idx, num_nodes in enumerate(layer_sizes):
                x_pos = layer_idx * layer_spacing
                y_positions = np.linspace(-(num_nodes - 1) / 2, (num_nodes - 1) / 2, num_nodes) * 1.5
                
                for node_idx, y in enumerate(y_positions):
                    activation_value = frame_activations[layer_idx][node_idx]
                    color = plt.cm.RdYlBu_r(activation_value)
                    
                    circle = plt.Circle((x_pos, y), 0.25, color=color, ec="black", linewidth=2.5, zorder=5, alpha=0.8)
                    ax_anim.add_patch(circle)
                    
                    ax_anim.text(x_pos, y, f"{activation_value:.2f}", ha="center", va="center", 
                               fontsize=8, fontweight="bold", 
                               color="white" if activation_value > 0.5 else "black", zorder=6)
            
            # Redraw connections
            for layer_idx in range(len(layer_sizes) - 1):
                from_positions = [(layer_idx * layer_spacing, y) 
                                for y in np.linspace(-(layer_sizes[layer_idx] - 1) / 2, 
                                                    (layer_sizes[layer_idx] - 1) / 2, 
                                                    layer_sizes[layer_idx]) * 1.5]
                to_positions = [((layer_idx + 1) * layer_spacing, y) 
                              for y in np.linspace(-(layer_sizes[layer_idx + 1] - 1) / 2, 
                                                  (layer_sizes[layer_idx + 1] - 1) / 2, 
                                                  layer_sizes[layer_idx + 1]) * 1.5]
                
                np.random.seed(layer_idx)
                weights = np.random.randn(len(from_positions), len(to_positions))
                weights = np.abs(weights)
                weights = weights / weights.max()
                
                for from_idx, (x1, y1) in enumerate(from_positions):
                    for to_idx, (x2, y2) in enumerate(to_positions):
                        weight_strength = weights[from_idx, to_idx]
                        ax_anim.plot([x1, x2], [y1, y2], 
                                   color=plt.cm.viridis(weight_strength),
                                   linewidth=weight_strength * 3 + 0.5,
                                   alpha=0.4,
                                   zorder=1)
            
            ax_anim.set_xlim(-1, (num_layers - 1) * layer_spacing + 1)
            ax_anim.set_ylim(-(max_nodes) * 1, (max_nodes) * 1)
            ax_anim.set_aspect('equal')
            ax_anim.axis('off')
            ax_anim.set_title(f"Data Flow Animation - Frame {frame_idx + 1}", 
                            fontsize=14, fontweight="bold", pad=20)
            
            return ax_anim,
        
        # Create gif
        anim = FuncAnimation(fig_anim, animate_frame, frames=len(frames_data), 
                           interval=int(1000 / steps), blit=False)
        
        tmp_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        tmp_file.close()
        anim.save(tmp_file.name, writer=PillowWriter(fps=steps))
        plt.close(fig_anim)
        
        with open(tmp_file.name, "rb") as f:
            st.image(f.read(), width='stretch', caption="Data propagating through network layers")
        
        os.unlink(tmp_file.name)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FORWARD PROPAGATION
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "🔄 Forward Propagation":
    st.header("🔄 Forward Propagation: Layer-by-Layer Computation")
    
    st.markdown("""
**Forward Propagation** is the process where data flows through the network:
- **Input** → multiply by **weights** → add **bias** → pass to **activation** → **Output**

Each layer transforms the input through: $\\mathbf{a}^{(l)} = \\sigma(\\mathbf{W}^{(l)} \\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)})$
    """)
    
    # Interactive parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        input_size = st.slider("Input Features", 2, 10, 3)
    with col2:
        hidden_size = st.slider("Hidden Units", 2, 10, 4)
    with col3:
        activation = st.selectbox("Activation", ["ReLU", "Sigmoid", "Tanh"])
    
    # Generate random input
    np.random.seed(42)
    X_input = np.random.randn(1, input_size)
    W = np.random.randn(input_size, hidden_size) * 0.5
    b = np.random.randn(1, hidden_size) * 0.1
    
    # Forward pass
    Z = np.dot(X_input, W) + b
    
    if activation == "ReLU":
        A = np.maximum(0, Z)
        activation_fn = lambda x: np.maximum(0, x)
    elif activation == "Sigmoid":
        A = 1 / (1 + np.exp(-Z))
        activation_fn = lambda x: 1 / (1 + np.exp(-x))
    else:  # Tanh
        A = np.tanh(Z)
        activation_fn = lambda x: np.tanh(x)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Input features
    ax = axes[0, 0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, input_size))
    bars1 = ax.bar(range(input_size), X_input[0], color=colors, edgecolor="black", linewidth=2)
    ax.set_title("Step 1: Input Features", fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Value")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(X_input[0]):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    
    # Plot 2: Z = W*X + b (weighted sum + bias)
    ax = axes[0, 1]
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, hidden_size))
    bars2 = ax.bar(range(hidden_size), Z[0], color=colors, edgecolor="black", linewidth=2)
    ax.set_title(f"Step 2: Z = W·X + b (Weighted Sum + Bias)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hidden Unit Index")
    ax.set_ylabel("Pre-Activation Value (Z)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(Z[0]):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    
    # Plot 3: Activation function curve
    ax = axes[1, 0]
    x_range = np.linspace(-3, 3, 100)
    y_range = activation_fn(x_range)
    ax.plot(x_range, y_range, "b-", linewidth=2.5, label=activation)
    ax.scatter(Z[0], A[0], color="red", s=100, zorder=5, edgecolor="darkred", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(f"Step 3: Apply {activation} Activation Function", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pre-Activation (Z)")
    ax.set_ylabel("Activated Output (A)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 4: Final output
    ax = axes[1, 1]
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, hidden_size))
    bars4 = ax.bar(range(hidden_size), A[0], color=colors, edgecolor="black", linewidth=2)
    ax.set_title("Step 4: Output (Activated Values)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hidden Unit Index")
    ax.set_ylabel("Activation Value (A)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(A[0]):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    st.image(fig_to_pil(fig), width="stretch")
    plt.close(fig)
    
    # Show computation details
    with st.expander("📊 Detailed Computation"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Input (X):**")
            st.write(X_input)
            st.markdown("**Weights (W):**")
            st.write(W)
        with col2:
            st.markdown("**Bias (b):**")
            st.write(b)
            st.markdown(f"**Z = W·X + b:**")
            st.write(Z)
        
        st.markdown(f"**A = {activation}(Z):**")
        st.write(A)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "⚡ Activation Functions":
    st.header("⚡ Activation Functions: Non-Linearity Magic")
    
    st.markdown("""
**Activation functions** introduce non-linearity, allowing networks to learn complex patterns.
Without them, stacking layers would just be a linear transformation!
    """)
    
    activations = {
        "ReLU": lambda x: np.maximum(0, x),
        "Sigmoid": lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
        "Tanh": lambda x: np.tanh(x),
        "Leaky ReLU": lambda x: np.where(x > 0, x, 0.01 * x),
        "ELU": lambda x: np.where(x > 0, x, 0.1 * (np.exp(x) - 1)),
    }
    
    derivatives = {
        "ReLU": lambda x: np.where(x > 0, 1, 0),
        "Sigmoid": lambda x: (1 / (1 + np.exp(-np.clip(x, -500, 500)))) * (1 - (1 / (1 + np.exp(-np.clip(x, -500, 500))))),
        "Tanh": lambda x: 1 - np.tanh(x) ** 2,
        "Leaky ReLU": lambda x: np.where(x > 0, 1, 0.01),
        "ELU": lambda x: np.where(x > 0, 1, 0.1 * np.exp(x)),
    }
    
    col1, col2 = st.columns(2)
    with col1:
        selected_activation = st.selectbox("Choose Activation:", list(activations.keys()))
    with col2:
        show_derivative = st.checkbox("Show Derivative", value=False)
    
    x_range = np.linspace(-5, 5, 200)
    y_range = activations[selected_activation](x_range)
    dy_range = derivatives[selected_activation](x_range)
    
    fig, axes = plt.subplots(1, 2 if show_derivative else 1, figsize=(14, 5))
    
    if not show_derivative:
        axes = [axes]
    
    # Function plot
    ax = axes[0]
    ax.plot(x_range, y_range, "b-", linewidth=3, label=selected_activation)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Input (x)", fontsize=11)
    ax.set_ylabel("Output", fontsize=11)
    ax.set_title(f"{selected_activation} Activation Function", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.fill_between(x_range, y_range, alpha=0.3)
    
    # Derivative plot
    if show_derivative:
        ax = axes[1]
        ax.plot(x_range, dy_range, "r-", linewidth=3, label=f"d/dx {selected_activation}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Input (x)", fontsize=11)
        ax.set_ylabel("Gradient", fontsize=11)
        ax.set_title(f"Derivative: d/dx {selected_activation}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.fill_between(x_range, dy_range, alpha=0.3, color="red")
    
    plt.tight_layout()
    st.image(fig_to_pil(fig), width="stretch")
    plt.close(fig)
    
    # Comparison and properties
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📋 Activation Properties")
        properties = {
            "ReLU": "Fast, sparse, no vanishing gradient in positive region",
            "Sigmoid": "Smooth, differentiable, prone to vanishing gradients",
            "Tanh": "Zero-centered, stronger gradients than sigmoid",
            "Leaky ReLU": "Prevents dying ReLU problem, allows negative flow",
            "ELU": "Smooth negative region, reduces bias shift",
        }
        st.write(f"**{selected_activation}:** {properties[selected_activation]}")
    
    with col2:
        st.markdown("### 🎯 When to Use")
        usage = {
            "ReLU": "Hidden layers (most popular choice)",
            "Sigmoid": "Binary classification output layer",
            "Tanh": "Hidden layers, RNNs, centered inputs",
            "Leaky ReLU": "Deep networks, replacement for ReLU",
            "ELU": "Deeper networks for better convergence",
        }
        st.write(f"**{selected_activation}:** {usage[selected_activation]}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: LOSS CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "📉 Loss Calculation":
    st.header("📉 Loss Calculation: Measuring Error")
    
    st.markdown("""
**Loss** measures how far our predictions are from the true values.
The network's goal is to **minimize this loss** through learning.
    """)
    
    loss_type = st.selectbox("Choose Loss Function:", ["Mean Squared Error (MSE)", "Cross Entropy", "MAE"])
    
    np.random.seed(42)
    y_true = np.array([1, 0, 1, 0, 1, 0])
    
    if loss_type == "Mean Squared Error (MSE)":
        y_pred_range = np.linspace(0, 1, 100)
        losses = []
        for y_p in y_pred_range:
            y_p_batch = np.full_like(y_true, y_p, dtype=float)
            loss = np.mean((y_true - y_p_batch) ** 2)
            losses.append(loss)
        losses = np.array(losses)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        ax = axes[0]
        ax.plot(y_pred_range, losses, "b-", linewidth=3)
        ax.fill_between(y_pred_range, losses, alpha=0.3)
        ax.set_xlabel("Predicted Value", fontsize=11)
        ax.set_ylabel("Loss (MSE)", fontsize=11)
        ax.set_title("Mean Squared Error Loss", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        
        # Predictions vs True
        ax = axes[1]
        x_pos = np.arange(len(y_true))
        width = 0.35
        bars1 = ax.bar(x_pos - width/2, y_true, width, label="True", color="green", alpha=0.7, edgecolor="black")
        
        y_pred = np.array([0.9, 0.1, 0.85, 0.2, 0.95, 0.05])
        bars2 = ax.bar(x_pos + width/2, y_pred, width, label="Predicted", color="red", alpha=0.7, edgecolor="black")
        
        ax.set_ylabel("Probability", fontsize=11)
        ax.set_title("True vs Predicted Values", fontsize=13, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Sample {i}" for i in range(len(y_true))])
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(axis="y", alpha=0.3)
        
        mse_loss = np.mean((y_true - y_pred) ** 2)
        ax.text(0.5, 0.95, f"MSE Loss = {mse_loss:.4f}", transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    elif loss_type == "Cross Entropy":
        y_pred_range = np.linspace(0.001, 0.999, 100)
        losses = []
        for y_p in y_pred_range:
            loss = -np.mean(y_true * np.log(y_p) + (1 - y_true) * np.log(1 - y_p))
            losses.append(loss)
        losses = np.array(losses)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        ax = axes[0]
        ax.plot(y_pred_range, losses, "b-", linewidth=3)
        ax.fill_between(y_pred_range, losses, alpha=0.3)
        ax.set_xlabel("Predicted Probability", fontsize=11)
        ax.set_ylabel("Loss (Cross Entropy)", fontsize=11)
        ax.set_title("Cross Entropy Loss (Binary Classification)", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        
        # Predictions vs True
        ax = axes[1]
        x_pos = np.arange(len(y_true))
        width = 0.35
        bars1 = ax.bar(x_pos - width/2, y_true, width, label="True", color="green", alpha=0.7, edgecolor="black")
        
        y_pred = np.array([0.9, 0.1, 0.85, 0.2, 0.95, 0.05])
        bars2 = ax.bar(x_pos + width/2, y_pred, width, label="Predicted", color="red", alpha=0.7, edgecolor="black")
        
        ax.set_ylabel("Probability", fontsize=11)
        ax.set_title("True vs Predicted Probabilities", fontsize=13, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Sample {i}" for i in range(len(y_true))])
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(axis="y", alpha=0.3)
        
        ce_loss = -np.mean(y_true * np.log(y_pred + 1e-10) + (1 - y_true) * np.log(1 - y_pred + 1e-10))
        ax.text(0.5, 0.95, f"Cross Entropy Loss = {ce_loss:.4f}", transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    else:  # MAE
        y_pred = np.array([0.9, 0.1, 0.85, 0.2, 0.95, 0.05])
        mae = np.mean(np.abs(y_true - y_pred))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        ax = axes[0]
        y_pred_range = np.linspace(0, 1, 100)
        losses = []
        for y_p in y_pred_range:
            y_p_batch = np.full_like(y_true, y_p, dtype=float)
            loss = np.mean(np.abs(y_true - y_p_batch))
            losses.append(loss)
        losses = np.array(losses)
        
        ax.plot(y_pred_range, losses, "b-", linewidth=3)
        ax.fill_between(y_pred_range, losses, alpha=0.3)
        ax.set_xlabel("Predicted Value", fontsize=11)
        ax.set_ylabel("Loss (MAE)", fontsize=11)
        ax.set_title("Mean Absolute Error Loss", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        
        # Predictions vs True
        ax = axes[1]
        x_pos = np.arange(len(y_true))
        width = 0.35
        bars1 = ax.bar(x_pos - width/2, y_true, width, label="True", color="green", alpha=0.7, edgecolor="black")
        bars2 = ax.bar(x_pos + width/2, y_pred, width, label="Predicted", color="red", alpha=0.7, edgecolor="black")
        
        ax.set_ylabel("Value", fontsize=11)
        ax.set_title("True vs Predicted Values", fontsize=13, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Sample {i}" for i in range(len(y_true))])
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(axis="y", alpha=0.3)
        
        ax.text(0.5, 0.95, f"MAE Loss = {mae:.4f}", transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    st.image(fig_to_pil(fig), width="stretch")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: BACKPROPAGATION
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "🔙 Backpropagation":
    st.header("🔙 Backpropagation: Computing Gradients")
    
    st.markdown("""
**Backpropagation** is how we compute gradients (how much each weight contributed to the error).
Using the **chain rule**, we propagate the error backwards through the network.

Formula: $\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial a} \\cdot \\frac{\\partial a}{\\partial z} \\cdot \\frac{\\partial z}{\\partial w}$
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        depth = st.slider("Network Depth (layers)", 2, 5, 3)
    with col2:
        show_gradients = st.checkbox("Show Gradient Magnitudes", value=True)
    
    # Create network visualization showing backprop
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Network structure
    layers = [3, 4, 4, 4, 1][:depth+1]
    layer_distance = 2
    neuron_distance = 1.5
    
    # Draw neurons
    neuron_positions = {}
    colors_forward = plt.cm.Blues(np.linspace(0.3, 0.9, depth))
    colors_backward = plt.cm.Reds(np.linspace(0.3, 0.9, depth))
    
    for layer_idx, num_neurons in enumerate(layers):
        y_positions = np.linspace(0, (num_neurons - 1) * neuron_distance, num_neurons)
        x_pos = layer_idx * layer_distance
        neuron_positions[layer_idx] = [(x_pos, y) for y in y_positions]
        
        for y_pos in y_positions:
            circle = plt.Circle((x_pos, y_pos), 0.25, color="lightblue", ec="black", linewidth=2, zorder=3)
            ax.add_patch(circle)
    
    # Draw connections with gradient flow
    for layer_idx in range(len(layers) - 1):
        for from_idx, (x1, y1) in enumerate(neuron_positions[layer_idx]):
            for to_idx, (x2, y2) in enumerate(neuron_positions[layer_idx + 1]):
                # Forward pass (blue)
                ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.3, linewidth=1, zorder=1)
    
    if show_gradients:
        # Overlay backward pass (red) for a few connections
        for layer_idx in range(len(layers) - 1):
            sample_from = min(1, len(neuron_positions[layer_idx]) - 1)
            sample_to = min(1, len(neuron_positions[layer_idx + 1]) - 1)
            x1, y1 = neuron_positions[layer_idx][sample_from]
            x2, y2 = neuron_positions[layer_idx + 1][sample_to]
            ax.annotate("", xy=(x1, y1), xytext=(x2, y2),
                       arrowprops=dict(arrowstyle="<-", color="red", lw=2.5, alpha=0.7))
    
    # Labels
    ax.text(-0.3, -2, "INPUT", fontsize=11, fontweight="bold", ha="center")
    for i in range(1, len(layers) - 1):
        ax.text(i * layer_distance, -2, f"HIDDEN {i}", fontsize=11, fontweight="bold", ha="center")
    ax.text((len(layers) - 1) * layer_distance, -2, "OUTPUT", fontsize=11, fontweight="bold", ha="center")
    
    ax.set_xlim(-1, (len(layers) - 1) * layer_distance + 1)
    ax.set_ylim(-2.5, max([max([y for _, y in pos]) for pos in neuron_positions.values()]) + 1)
    ax.axis("off")
    
    # Title and legend
    title_text = "Forward Pass (→) and Backpropagation (←)" if show_gradients else "Forward Pass Only (→)"
    ax.set_title(title_text, fontsize=14, fontweight="bold", pad=20)
    
    if show_gradients:
        ax.text(0.02, 0.98, "→ Forward: Input → Output\n← Backward: Gradients ← Loss",
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    st.image(fig_to_pil(fig), width="stretch")
    plt.close(fig)
    
    # Backprop steps explanation
    st.markdown("### 📋 Backpropagation Steps")
    
    steps = [
        "1️⃣ **Forward Pass**: Compute activation in each layer (a¹, a², ..., aᴸ)",
        "2️⃣ **Compute Loss**: Calculate L from final output vs true labels",
        "3️⃣ **Output Gradient**: ∂L/∂aᴸ (error in output layer)",
        "4️⃣ **Hidden Gradients**: Use chain rule to propagate backwards: ∂L/∂aˡ⁻¹",
        "5️⃣ **Weight Gradients**: ∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W",
        "6️⃣ **Update Weights**: W ← W - α · ∂L/∂W (gradient descent step)"
    ]
    
    for step in steps:
        st.write(step)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: GRADIENT DESCENT VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

else:  # Gradient Descent
    st.header("🎯 Gradient Descent: 3 Optimization Strategies")
    
    st.markdown("""
After computing gradients via backpropagation, we use **gradient descent** to update weights.
There are 3 main variants that differ in how many samples we use per update:
    """)
    
    # Select visualization type
    viz_type = st.radio(
        "Choose Visualization:",
        ["📊 Loss Curves Comparison", "🗺️ 2D Loss Surface", "📈 Step-by-Step Animation"],
        horizontal=True
    )
    
    # Loss function (simple 2D for visualization)
    def loss_function_2d(w1, w2):
        return (w1 - 0.5) ** 2 + (w2 + 0.3) ** 2 + 0.3 * w1 * w2
    
    def loss_function_1d(iterations, noise_level=0):
        x = np.linspace(0, 10, iterations)
        base_loss = (x - 5) ** 2 / 10 + 2
        noise = np.random.randn(iterations) * noise_level
        return base_loss + noise
    
    if viz_type == "📊 Loss Curves Comparison":
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Parameters
        learning_rate = 0.1
        epochs = 50
        batch_size = 5
        total_samples = 100
        
        iterations = np.arange(epochs)
        
        # Batch GD (update per epoch)
        sgd_loss = loss_function_1d(epochs, noise_level=1.5)
        
        # Mini-batch GD (less noisy)
        mini_batch_loss = loss_function_1d(epochs, noise_level=0.8)
        
        # Batch GD (smooth)
        batch_loss = loss_function_1d(epochs, noise_level=0.0)
        
        # SGD
        ax = axes[0]
        ax.plot(iterations, sgd_loss, "r.-", linewidth=2, markersize=8, label="SGD Loss")
        ax.fill_between(iterations, sgd_loss, alpha=0.3, color="red")
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("Stochastic GD (SGD)\n1 sample per update", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 10])
        
        # Mini-batch
        ax = axes[1]
        ax.plot(iterations, mini_batch_loss, "g.-", linewidth=2, markersize=8, label="Mini-Batch Loss")
        ax.fill_between(iterations, mini_batch_loss, alpha=0.3, color="green")
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title(f"Mini-Batch GD\n{batch_size} samples per update", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 10])
        
        # Batch
        ax = axes[2]
        ax.plot(iterations, batch_loss, "b.-", linewidth=2, markersize=8, label="Batch Loss")
        ax.fill_between(iterations, batch_loss, alpha=0.3, color="blue")
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title(f"Batch GD\nAll {total_samples} samples per update", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 10])
        
        plt.tight_layout()
        st.image(fig_to_pil(fig), width="stretch")
        plt.close(fig)
        
        # Comparison table
        st.markdown("### 📋 Comparison")
        comparison_data = {
            "Aspect": ["Update Frequency", "Gradient Noise", "Convergence", "Memory", "Escaping Local Minima"],
            "SGD": ["Very High", "Very High", "Noisy", "Low", "↑ Good"],
            "Mini-Batch": ["Medium", "Medium", "Moderate", "Medium", "⚖️ Balanced"],
            "Batch": ["Low (1/epoch)", "None", "Smooth", "High", "↓ Poor"]
        }
        st.dataframe(comparison_data, width='stretch')
    
    elif viz_type == "🗺️ 2D Loss Surface":
        w1_range = np.linspace(-1, 2, 100)
        w2_range = np.linspace(-1.5, 1, 100)
        W1, W2 = np.meshgrid(w1_range, w2_range)
        Z = loss_function_2d(W1, W2)
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Starting point
        start_w1, start_w2 = -0.5, 0.8
        
        methods = [
            ("SGD (noisy)", 0.1, "red"),
            ("Mini-Batch", 0.15, "green"),
            ("Batch (smooth)", 0.15, "blue"),
        ]
        
        for idx, (name, lr, color) in enumerate(methods):
            ax = axes[idx]
            
            # Contour plot
            contour = ax.contour(W1, W2, Z, levels=15, alpha=0.6, cmap="viridis")
            ax.clabel(contour, inline=True, fontsize=8)
            ax.contourf(W1, W2, Z, levels=15, alpha=0.3, cmap="viridis")
            
            # Mark optimal point
            ax.plot(0.5, -0.3, "g*", markersize=20, label="Optimal", zorder=5)
            
            # Simulate optimization path
            w1, w2 = start_w1, start_w2
            path_w1, path_w2 = [w1], [w2]
            
            for _ in range(20):
                dw1 = 2 * (w1 - 0.5) + 0.3 * w2
                dw2 = 2 * (w2 + 0.3) + 0.3 * w1
                
                if "SGD" in name:
                    dw1 += np.random.randn() * 0.5
                    dw2 += np.random.randn() * 0.5
                
                w1 -= lr * dw1
                w2 -= lr * dw2
                path_w1.append(w1)
                path_w2.append(w2)
            
            ax.plot(path_w1, path_w2, f"{color[0]}-", linewidth=2, markersize=6, marker="o", alpha=0.7)
            ax.plot(start_w1, start_w2, "ko", markersize=10, label="Start", zorder=5)
            
            ax.set_xlabel("Weight 1 (w₁)", fontsize=11)
            ax.set_ylabel("Weight 2 (w₂)", fontsize=11)
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        st.image(fig_to_pil(fig), width="stretch")
        plt.close(fig)
    
    else:  # Animation
        st.markdown("### 🎬 Step-by-Step Gradient Descent")
        
        method = st.selectbox("Select Method:", ["SGD", "Mini-Batch", "Batch GD"])
        
        # Parameters
        if method == "SGD":
            noise = 1.0
            title_info = "Stochastic GD (1 sample): Noisy but can escape local minima"
        elif method == "Mini-Batch":
            noise = 0.5
            title_info = "Mini-Batch GD (5 samples): Balanced approach"
        else:
            noise = 0.0
            title_info = "Batch GD (all samples): Smooth path to minima"
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Generate loss landscape
        iterations_range = np.arange(40)
        loss_curve = loss_function_1d(40, noise_level=noise)
        
        ax.plot(iterations_range, loss_curve, f"{'r' if method == 'SGD' else 'g' if method == 'Mini-Batch' else 'b'}-", 
                linewidth=3, label="Loss trajectory")
        ax.fill_between(iterations_range, loss_curve, alpha=0.3, 
                        color='red' if method == 'SGD' else 'green' if method == 'Mini-Batch' else 'blue')
        ax.set_xlabel("Training Iteration", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"{title_info}", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        
        # Add annotations
        ax.scatter([0], [loss_curve[0]], color="green", s=200, marker="o", label="Start", zorder=5, edgecolor="black")
        ax.scatter([len(loss_curve)-1], [loss_curve[-1]], color="red", s=200, marker="*", label="End", zorder=5, edgecolor="black")
        
        ax.legend(fontsize=11, loc="upper right")
        
        plt.tight_layout()
        st.image(fig_to_pil(fig), width="stretch")
        plt.close(fig)
        
        # Key points
        st.markdown("### 🔑 Key Takeaways")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**SGD**")
            st.write("• Frequent updates\n• High noise\n• Can escape local minima\n• Hard to parallelize")
        
        with col2:
            st.markdown("**Mini-Batch** ⭐")
            st.write("• Best trade-off\n• Moderate noise\n• Good parallelization\n• Preferred in practice")
        
        with col3:
            st.markdown("**Batch GD**")
            st.write("• Smooth convergence\n• No noise\n• Can get stuck\n• High memory usage")

st.markdown("---")
st.info("💡 **Tip**: Interact with different sections to understand how neural networks learn!")
