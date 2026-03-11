import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
import plotly.graph_objects as go
import io
import time
from typing import Optional

# Helper function to convert matplotlib figure to PIL image
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return Image.open(buf)

st.set_page_config(page_title="Neural Network Training", layout="wide")
st.title("🧠 Neural Network: Complete Training Workflow")

# Initialize session state
if 'weights' not in st.session_state:
    st.session_state.weights = None
if 'biases' not in st.session_state:
    st.session_state.biases = None

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: INPUT FEATURE ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

st.header("📊 Step 1: Input Feature Animation")
st.markdown("Watch how input features are prepared and fed into the neural network.")

col1, col2, col3 = st.columns(3)
with col1:
    num_features = st.slider("Number of Input Features", 2, 6, 3)
with col2:
    sample_type = st.selectbox("Sample Type", ["Random", "Pattern 1", "Pattern 2"])
with col3:
    show_input_animation = st.checkbox("Show Step-by-Step Animation", value=True)

# Generate input features
np.random.seed(42)
if sample_type == "Random":
    input_features = np.random.rand(num_features)
elif sample_type == "Pattern 1":
    input_features = np.sin(np.linspace(0, np.pi, num_features))
else:  # Pattern 2
    input_features = np.cos(np.linspace(0, 2*np.pi, num_features))

# Visualize input features
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Input feature values
axes[0].bar(range(num_features), input_features, color=plt.cm.viridis(np.linspace(0, 1, num_features)))
axes[0].set_xlabel("Feature Index")
axes[0].set_ylabel("Feature Value")
axes[0].set_title("Input Features")
axes[0].set_ylim([0, 1.1])
for i, v in enumerate(input_features):
    axes[0].text(i, v + 0.05, f"{v:.3f}", ha="center", fontsize=9)

# Feature distribution animation concept
for i in range(num_features):
    axes[1].scatter([i], [input_features[i]], s=500, alpha=0.6, label=f"Feature {i}")
axes[1].set_xlabel("Feature Index")
axes[1].set_ylabel("Normalized Value")
axes[1].set_title("Feature Distribution")
axes[1].set_ylim([0, 1.1])
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
st.image(fig_to_pil(fig), width='stretch')
plt.close(fig)

st.success(f"✅ {num_features} input features loaded and ready!")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FORWARD PROPAGATION WITH ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

st.header("🔄 Step 2: Forward Propagation - Data Flow Animation")
st.markdown("Watch how input data flows through the network, gets transformed by weights, biases, and activation functions.")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    num_hidden_layers = st.slider("Number of Hidden Layers", 1, 3, 2)
with col2:
    hidden_units = st.slider("Hidden Units per Layer", 2, 8, 4)
with col3:
    activation_func = st.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh"])
with col4:
    anim_speed = st.slider("Animation Speed", 1, 10, 5, help="Higher = faster")
with col5:
    show_forward_animation = st.checkbox("Animate Forward Pass", value=False)

# Convert speed slider to delay: speed 1 → 3s, speed 10 → 0.2s
anim_delay = 3.0 / anim_speed

# Initialize weights and biases if not already done or if layer configuration changed
np.random.seed(42)
layer_sizes = [num_features] + [hidden_units] * num_hidden_layers + [1]

# Reset weights if configuration changed
def _needs_reset():
    if st.session_state.weights is None:
        return True
    if len(st.session_state.weights) != len(layer_sizes) - 1:
        return True
    for i, w in enumerate(st.session_state.weights):
        if w.shape != (layer_sizes[i], layer_sizes[i + 1]):
            return True
    return False

if _needs_reset():
    st.session_state.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5 
                                for i in range(len(layer_sizes)-1)]
    st.session_state.biases = [np.random.randn(layer_sizes[i+1]) * 0.1 
                               for i in range(len(layer_sizes)-1)]

# Activation function
def activate(z, func):
    if func == "ReLU":
        return np.maximum(0, z)
    elif func == "Sigmoid":
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    else:  # Tanh
        return np.tanh(z)

# Forward pass
st.subheader("Forward Pass Visualization")

activations = [input_features.copy()]
z_values = []

current = input_features.copy()
for i, (w, b) in enumerate(zip(st.session_state.weights, st.session_state.biases)):
    z = np.dot(current, w) + b
    z_values.append(z)
    a = activate(z, activation_func)
    activations.append(a)
    current = a

# Create animated network visualization showing data flow
if show_forward_animation:

    # helper to build math breakdown text for a given layer transition
    def get_math_text(layer_idx):
        """Return a formatted string showing the math for layer_idx (1-based destination)."""
        src = layer_idx - 1
        w = st.session_state.weights[src]
        b = st.session_state.biases[src]
        a_in = activations[src]
        z = z_values[src]
        a_out = activations[layer_idx]

        layer_names_list = ["Input"] + [f"Hidden {i+1}" for i in range(num_hidden_layers)] + ["Output"]
        src_name = layer_names_list[src]
        dst_name = layer_names_list[layer_idx]

        lines = []
        lines.append(f"### 🔢 {src_name} → {dst_name}")
        lines.append("")

        # show input vector
        a_in_str = ", ".join(f"{v:.3f}" for v in a_in)
        lines.append(f"**Input (a{src}):** [{a_in_str}]")
        lines.append("")

        # show weight matrix
        lines.append(f"**Weights W{src}** ({w.shape[0]}×{w.shape[1]}):")
        lines.append("```")
        for row in w:
            lines.append("  [" + "  ".join(f"{v:+.3f}" for v in row) + "]")
        lines.append("```")

        # show bias vector
        b_str = ", ".join(f"{v:+.3f}" for v in b)
        lines.append(f"**Bias b{src}:** [{b_str}]")
        lines.append("")

        # show z = Wa + b calculation
        lines.append(f"**z = a{src} · W{src} + b{src}:**")
        lines.append("```")
        for j in range(len(z)):
            terms = " + ".join(f"({a_in[k]:.3f}×{w[k, j]:+.3f})" for k in range(len(a_in)))
            lines.append(f"  z[{j}] = {terms} + ({b[j]:+.3f})")
            lines.append(f"       = {z[j]:.4f}")
        lines.append("```")

        # show activation
        z_str = ", ".join(f"{v:.4f}" for v in z)
        a_out_str = ", ".join(f"{v:.4f}" for v in a_out)
        lines.append(f"**{activation_func}([{z_str}]):**")
        lines.append(f"**= [{a_out_str}]**")

        return "\n".join(lines)

    # helper to draw a single frame with optional highlight of current layer
    def draw_network(highlight_layer: Optional[int] = None):
        fig = plt.figure(figsize=(20,20), facecolor='#1a1a1a')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0d0d0d')

        # layout parameters
        layer_spacing = 3
        max_neurons = max(layer_sizes)
        neuron_radius = 0.3

        # calculate positions
        neuron_positions = {}
        for layer_idx, num_neurons in enumerate(layer_sizes):
            x_pos = layer_idx * layer_spacing
            y_positions = np.linspace(-(num_neurons - 1) / 2, (num_neurons - 1) / 2, num_neurons) * 1.5
            neuron_positions[layer_idx] = [(x_pos, y) for y in y_positions]

        # draw connections
        for layer_idx in range(len(layer_sizes) - 1):
            from_positions = neuron_positions[layer_idx]
            to_positions = neuron_positions[layer_idx + 1]
            weights = np.abs(st.session_state.weights[layer_idx])
            weights_norm = weights / (weights.max() + 1e-8)

            for from_idx, (x1, y1) in enumerate(from_positions):
                for to_idx, (x2, y2) in enumerate(to_positions):
                    weight_strength = weights_norm[from_idx, to_idx]
                    color = '#ff6b6b' if layer_idx % 2 == 0 else '#51cf66'
                    alpha = 0.8 if highlight_layer == layer_idx + 1 else 0.25
                    ax.plot([x1, x2], [y1, y2],
                           color=color,
                           linewidth=weight_strength * 3 + 0.5,
                           alpha=alpha,
                           zorder=1)

            # show weight values on highlighted edges
            if highlight_layer == layer_idx + 1:
                w = st.session_state.weights[layer_idx]
                for from_idx, (x1, y1) in enumerate(from_positions):
                    for to_idx, (x2, y2) in enumerate(to_positions):
                        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                        ax.text(mx, my, f"{w[from_idx, to_idx]:.2f}",
                                ha='center', va='center', fontsize=5,
                                color='#ffdd57', alpha=0.9, zorder=4,
                                bbox=dict(boxstyle='round,pad=0.1', facecolor='#000000', alpha=0.6, edgecolor='none'))

        # draw neurons
        for layer_idx, num_neurons in enumerate(layer_sizes):
            activation_vals = activations[layer_idx]

            for neuron_idx, (x, y) in enumerate(neuron_positions[layer_idx]):
                if neuron_idx < len(activation_vals):
                    intensity = np.clip(activation_vals[neuron_idx], 0, 1)
                else:
                    intensity = 0

                color_intensity = intensity
                if intensity > 0:
                    neuron_color = plt.cm.YlOrRd(0.3 + 0.7 * color_intensity)
                else:
                    neuron_color = '#333333'

                circle = plt.Circle((x, y), neuron_radius,
                                    color=neuron_color,
                                    ec='#ffd93d' if highlight_layer == layer_idx else ('#ffffff' if intensity > 0.5 else '#666666'),
                                    linewidth=3 if highlight_layer == layer_idx else 2,
                                    zorder=5,
                                    alpha=0.9)
                ax.add_patch(circle)

                if intensity > 0.2:
                    glow = plt.Circle((x, y), neuron_radius * 1.5,
                                      color=neuron_color,
                                      alpha=0.3 if highlight_layer == layer_idx else 0.15,
                                      zorder=3)
                    ax.add_patch(glow)

                if neuron_idx < len(activation_vals):
                    ax.text(x, y, f"{activation_vals[neuron_idx]:.2f}",
                            ha='center', va='center',
                            fontsize=7, fontweight='bold',
                            color='white' if intensity > 0.5 else '#999999',
                            zorder=6)

        ax.set_xlim(-1, (len(layer_sizes) - 1) * layer_spacing + 1)
        ax.set_ylim(-(max_neurons * 1.5) - 1.5, max_neurons * 1.5 + 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        return fig

    # build and display animation frames (loops continuously)
    net_placeholder = st.empty()
    math_placeholder = st.empty()
    while True:
        for layer_idx in range(1, len(layer_sizes)):
            fig = draw_network(highlight_layer=layer_idx)
            net_placeholder.image(fig_to_pil(fig), width='stretch')
            plt.close(fig)
            math_placeholder.markdown(get_math_text(layer_idx))
            time.sleep(anim_delay)
        # show full network briefly before restarting
        fig = draw_network()
        net_placeholder.image(fig_to_pil(fig), width='stretch')
        plt.close(fig)
        math_placeholder.markdown("### ✅ Forward pass complete! Restarting...")
        time.sleep(anim_delay)

else:
    # Simple bar chart visualization
    num_layers_to_show = min(num_hidden_layers + 2, 4)
    fig, axes = plt.subplots(1, num_layers_to_show, figsize=(5*num_layers_to_show, 4))
    
    if num_layers_to_show == 1:
        axes = [axes]
    
    for layer_idx in range(num_layers_to_show):
        activation_vals = activations[layer_idx]
        colors = plt.cm.RdYlBu(activation_vals / (np.max(np.abs(activation_vals)) + 1e-8))
        
        axes[layer_idx].bar(range(len(activation_vals)), activation_vals, color=colors)
        axes[layer_idx].set_title(f"Layer {layer_idx}\n({len(activation_vals)} neurons)")
        axes[layer_idx].set_ylabel("Activation Value")
        axes[layer_idx].set_ylim([0, 1.1])
        axes[layer_idx].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(activation_vals):
            axes[layer_idx].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    
    plt.tight_layout()
    st.image(fig_to_pil(fig), width='stretch')
    plt.close(fig)

st.markdown(f"**✅ Forward pass complete!** Output value: `{activations[-1][0]:.4f}`")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: BACKPROPAGATION - BACKWARD FLOW + DECISION BOUNDARY
# ═══════════════════════════════════════════════════════════════════════════════

st.header("🔙 Step 3: Backpropagation — Gradient Flow & Decision Boundary")
st.markdown(
    "Watch gradients flow **backward** from loss through the network. "
    "As weights and biases update, see the decision boundary shift in real time."
)

col_bp1, col_bp2, col_bp3, col_bp4 = st.columns(4)
with col_bp1:
    target_value = st.radio("Target Class (y)", [0, 1], index=1, horizontal=True)
with col_bp2:
    learning_rate = st.slider("Learning Rate (α)", 0.001, 0.5, 0.05, step=0.005, format="%.3f")
with col_bp3:
    st.write("")
with col_bp4:
    show_backprop_animation = st.checkbox("Animate Backpropagation", value=False)

# ── helpers for backprop ─────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def activate_deriv(z, func):
    """Derivative of the activation function."""
    if func == "ReLU":
        return (z > 0).astype(float)
    elif func == "Sigmoid":
        s = sigmoid(z)
        return s * (1 - s)
    else:  # Tanh
        return 1 - np.tanh(z) ** 2

def forward_pass(weights, biases, x):
    """Run forward pass with given weights/biases, return activations and z values."""
    acts = [x.copy()]
    zs = []
    cur = x.copy()
    for w, b in zip(weights, biases):
        z = np.dot(cur, w) + b
        zs.append(z)
        a = activate(z, activation_func)
        acts.append(a)
        cur = a
    return acts, zs

def compute_gradients(weights, biases, acts, zs, target):
    """Compute dW and db for each layer via backprop (log loss)."""
    n_layers = len(weights)
    dWs = [None] * n_layers
    dbs = [None] * n_layers
    # output error for BCE: d(loss)/d(z) = a_out - target (when output uses sigmoid)
    a_out = np.clip(acts[-1], 1e-7, 1 - 1e-7)
    delta = (a_out - target)
    for i in range(n_layers - 1, -1, -1):
        a_prev = acts[i]
        dWs[i] = np.outer(a_prev, delta)
        dbs[i] = delta.copy()
        if i > 0:
            delta = np.dot(delta, weights[i].T) * activate_deriv(zs[i - 1], activation_func)
    return dWs, dbs

def decision_boundary_fig(weights, biases, title="Decision Boundary"):
    """Plot decision boundary using first 2 input dims over a grid, with sample data."""
    fig, ax = plt.subplots(figsize=(5, 5))
    grid_res = 80
    x_range = np.linspace(-1.5, 2.5, grid_res)
    y_range = np.linspace(-1.5, 2.5, grid_res)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # pad extra dims with zeros if network expects >2 features
    n_in = weights[0].shape[0]
    if n_in > 2:
        pad = np.zeros((grid_points.shape[0], n_in - 2))
        grid_points = np.concatenate([grid_points, pad], axis=1)
    # forward pass on grid
    cur = grid_points
    for w, b in zip(weights, biases):
        z = cur @ w + b
        cur = activate(z, activation_func)
    zz = cur.reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap='RdYlBu', alpha=0.8)
    ax.contour(xx, yy, zz, levels=[0.5], colors='white', linewidths=2)
    # scatter some reference points
    np.random.seed(7)
    pts_class0 = np.random.randn(15, 2) * 0.4 + np.array([0.5, 0.5])
    pts_class1 = np.random.randn(15, 2) * 0.4 + np.array([1.5, 1.5])
    ax.scatter(pts_class0[:, 0], pts_class0[:, 1], c='blue', edgecolors='white', s=60, zorder=5, label='Class 0')
    ax.scatter(pts_class1[:, 0], pts_class1[:, 1], c='red', edgecolors='white', s=60, zorder=5, label='Class 1')
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1.5, 2.5)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig

def draw_backprop_network(weights, biases, acts, grad_layer=None):
    """Draw network with backward-flow highlight (red arrows from output toward input)."""
    sizes = [weights[0].shape[0]] + [w.shape[1] for w in weights]
    fig = plt.figure(figsize=(20, 20), facecolor='#1a1a1a')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0d0d0d')
    layer_spacing = 3
    max_n = max(sizes)
    r = 0.3
    positions = {}
    for li, n in enumerate(sizes):
        x = li * layer_spacing
        ys = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * 1.5
        positions[li] = [(x, y) for y in ys]
    # draw connections
    for li in range(len(sizes) - 1):
        fp = positions[li]
        tp = positions[li + 1]
        w_abs = np.abs(weights[li])
        w_norm = w_abs / (w_abs.max() + 1e-8)
        for fi, (x1, y1) in enumerate(fp):
            for ti, (x2, y2) in enumerate(tp):
                is_grad = grad_layer is not None and li == len(sizes) - 2 - grad_layer
                color = '#ff4444' if is_grad else '#555555'
                alpha = 0.85 if is_grad else 0.25
                ax.annotate("", xy=(x1, y1), xytext=(x2, y2),
                            arrowprops=dict(arrowstyle='->', color=color,
                                            lw=w_norm[fi, ti] * 2.5 + 0.5, alpha=alpha))
    # draw neurons
    for li, n in enumerate(sizes):
        a_vals = acts[li]
        for ni, (x, y) in enumerate(positions[li]):
            intensity = float(np.clip(a_vals[ni], 0, 1)) if ni < len(a_vals) else 0
            nc = plt.cm.YlOrRd(0.3 + 0.7 * intensity) if intensity > 0 else '#333333'
            is_hl = grad_layer is not None and li == len(sizes) - 1 - grad_layer
            c = plt.Circle((x, y), r, color=nc,
                           ec='#ff4444' if is_hl else '#666666',
                           linewidth=3 if is_hl else 1.5, zorder=5, alpha=0.9)
            ax.add_patch(c)
            if ni < len(a_vals):
                ax.text(x, y, f"{a_vals[ni]:.2f}", ha='center', va='center',
                        fontsize=6, fontweight='bold',
                        color='white' if intensity > 0.5 else '#aaa', zorder=6)
    # arrow showing backward direction
    ax.annotate("← GRADIENTS FLOW", xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=10, color='#ff6b6b', fontweight='bold',
                ha='right', va='bottom')
    ax.set_xlim(-1, (len(sizes) - 1) * layer_spacing + 1)
    ax.set_ylim(-max_n * 1.5 - 1, max_n * 1.5 + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig

# ── run backprop animation ───────────────────────────────────────────────────

if show_backprop_animation:
    # deep copy weights/biases so animation doesn't permanently mutate session
    bp_weights = [w.copy() for w in st.session_state.weights]
    bp_biases = [b.copy() for b in st.session_state.biases]

    col_net, col_db = st.columns(2)
    with col_net:
        bp_net_ph = st.empty()
    with col_db:
        bp_db_ph = st.empty()
    bp_math_ph = st.empty()

    n_transitions = len(bp_weights)  # number of layer gaps

    while True:
        # --- compute current forward pass ---
        bp_acts, bp_zs = forward_pass(bp_weights, bp_biases, input_features)
        output_val = bp_acts[-1][0]
        out_clipped = np.clip(output_val, 1e-7, 1 - 1e-7)
        loss = -(target_value * np.log(out_clipped) + (1 - target_value) * np.log(1 - out_clipped))

        # --- compute gradients ---
        dWs, dbs = compute_gradients(bp_weights, bp_biases, bp_acts, bp_zs, np.array([target_value]))

        # --- animate backward layer by layer (output → input) ---
        for step in range(n_transitions):
            # step 0 = output layer, step n-1 = first hidden layer
            layer_from_end = step
            layer_idx = n_transitions - 1 - step  # index into weights list

            # draw network with gradient highlight
            fig_net = draw_backprop_network(bp_weights, bp_biases, bp_acts, grad_layer=layer_from_end)
            bp_net_ph.image(fig_to_pil(fig_net), width='stretch')
            plt.close(fig_net)

            # draw decision boundary
            fig_db = decision_boundary_fig(bp_weights, bp_biases,
                                           title=f"Decision Boundary  |  Loss: {loss:.4f}")
            bp_db_ph.image(fig_to_pil(fig_db), width='stretch')
            plt.close(fig_db)

            # math text
            w = bp_weights[layer_idx]
            b = bp_biases[layer_idx]
            dw = dWs[layer_idx]
            db = dbs[layer_idx]
            lines = []
            lines.append(f"### 🔙 Backprop — Layer {layer_idx} (→ Layer {layer_idx + 1})")
            lines.append(f"**Loss (Log Loss):** {loss:.6f}  |  **Output:** {output_val:.4f}  |  **Target:** {target_value}")
            lines.append("")
            lines.append(f"**∂L/∂W{layer_idx}:**")
            lines.append("```")
            for row in dw:
                lines.append("  [" + "  ".join(f"{v:+.4f}" for v in row) + "]")
            lines.append("```")
            lines.append(f"**∂L/∂b{layer_idx}:** [{', '.join(f'{v:+.4f}' for v in db)}]")
            lines.append("")
            lines.append(f"**W{layer_idx}_new = W{layer_idx} − α·∂L/∂W{layer_idx}**")
            w_new = w - learning_rate * dw
            lines.append("```")
            for r_old, r_new in zip(w, w_new):
                old_s = "  ".join(f"{v:+.3f}" for v in r_old)
                new_s = "  ".join(f"{v:+.3f}" for v in r_new)
                lines.append(f"  [{old_s}] → [{new_s}]")
            lines.append("```")
            b_new = b - learning_rate * db
            lines.append(f"**b{layer_idx}_new:** [{', '.join(f'{v:+.4f}' for v in b_new)}]")
            bp_math_ph.markdown("\n".join(lines))

            time.sleep(anim_delay)

        # --- apply gradient update ---
        for i in range(len(bp_weights)):
            bp_weights[i] = bp_weights[i] - learning_rate * dWs[i]
            bp_biases[i] = bp_biases[i] - learning_rate * dbs[i]

        # show updated decision boundary
        bp_acts_new, _ = forward_pass(bp_weights, bp_biases, input_features)
        new_out = np.clip(bp_acts_new[-1][0], 1e-7, 1 - 1e-7)
        new_loss = -(target_value * np.log(new_out) + (1 - target_value) * np.log(1 - new_out))
        fig_db = decision_boundary_fig(bp_weights, bp_biases,
                                       title=f"Updated Boundary  |  Loss: {loss:.4f} → {new_loss:.4f}")
        bp_db_ph.image(fig_to_pil(fig_db), width='stretch')
        plt.close(fig_db)
        bp_math_ph.markdown(f"### ✅ Weights updated!  Loss: **{loss:.4f} → {new_loss:.4f}**  — Restarting backprop...")
        time.sleep(anim_delay * 1.5)

else:
    st.info("☑️ Check **Animate Backpropagation** above to start the backward-flow animation.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: GRADIENT DESCENT — 3D OPTIMISATION LANDSCAPE
# ═══════════════════════════════════════════════════════════════════════════════

st.header("📉 Step 4: Gradient Descent — Optimization Landscape")
st.markdown(
    "Visualize how **Batch**, **Stochastic (SGD)**, and **Mini-Batch** gradient descent "
    "navigate a loss surface to reach the optimal point.  \n"
    "3D surface and 2D contour plots are shown side by side."
)

col_gd1, col_gd2, col_gd3, col_gd4 = st.columns(4)
with col_gd1:
    gd_type = st.selectbox(
        "Gradient Descent Type",
        ["Batch", "Stochastic (SGD)", "Mini-Batch"],
    )
with col_gd2:
    gd_lr = st.slider(
        "Learning Rate (GD)", 0.001, 0.3, 0.05,
        step=0.005, format="%.3f", key="gd_lr",
    )
with col_gd3:
    gd_steps = st.slider("Optimisation Steps", 10, 150, 50, key="gd_steps")
with col_gd4:
    show_gd_animation = st.checkbox("Animate Gradient Descent", value=False)

# ── Loss surface ─────────────────────────────────────────────────────────────
# Hilly landscape: quadratic bowl + spread-out sinusoidal bumps
# L(w1, w2) = 0.3*((w1-1)^2 + (w2-1)^2) + 3*sin²(0.8*w1) + 3*sin²(0.8*w2)

def gd_loss(w1, w2):
    base = 0.3 * ((w1 - 1.0) ** 2 + (w2 - 1.0) ** 2)
    hills = 3.0 * np.sin(0.8 * w1) ** 2 + 3.0 * np.sin(0.8 * w2) ** 2
    return base + hills


def gd_gradient(w1, w2):
    # ∂L/∂w1 = 0.6*(w1-1) + 2.4*sin(1.6*w1)
    # ∂L/∂w2 = 0.6*(w2-1) + 2.4*sin(1.6*w2)
    dw1 = 0.6 * (w1 - 1.0) + 2.4 * np.sin(1.6 * w1)
    dw2 = 0.6 * (w2 - 1.0) + 2.4 * np.sin(1.6 * w2)
    return np.array([dw1, dw2])


def generate_gd_path(kind, lr, n_steps, start=(-2.0, -2.0)):
    """Simulate an optimisation trajectory with optional noise."""
    rng = np.random.RandomState(42)
    w = np.array(start, dtype=float)
    trajectory = [w.copy()]
    for _ in range(n_steps):
        g = gd_gradient(w[0], w[1])
        if kind == "Stochastic (SGD)":
            g = g + rng.randn(2) * 2.0        # high variance
        elif kind == "Mini-Batch":
            g = g + rng.randn(2) * 0.7         # moderate variance
        w = w - lr * g
        trajectory.append(w.copy())
    return np.array(trajectory)


# Pre-compute path & mesh
gd_path = generate_gd_path(gd_type, gd_lr, gd_steps)

w1_grid = np.linspace(-3.5, 4.5, 80)
w2_grid = np.linspace(-3.5, 4.5, 80)
W1g, W2g = np.meshgrid(w1_grid, w2_grid)
Lg = gd_loss(W1g, W2g)


def _draw_gd_3d_plotly(idx):
    """Return an interactive Plotly 3D figure for frame *idx*."""
    sub = gd_path[: idx + 1]
    losses = gd_loss(sub[:, 0], sub[:, 1])

    fig = go.Figure()

    # Loss surface
    fig.add_trace(go.Surface(
        x=W1g, y=W2g, z=Lg,
        colorscale="Viridis", opacity=0.7,
        showscale=False,
        name="Loss Surface",
    ))

    # Trajectory path
    fig.add_trace(go.Scatter3d(
        x=sub[:, 0], y=sub[:, 1], z=losses,
        mode="lines+markers",
        marker=dict(size=4, color="red"),
        line=dict(color="red", width=4),
        name="GD Path",
    ))

    # Current point
    fig.add_trace(go.Scatter3d(
        x=[sub[-1, 0]], y=[sub[-1, 1]], z=[losses[-1]],
        mode="markers",
        marker=dict(size=10, color="red", symbol="circle",
                    line=dict(color="white", width=2)),
        name="Current",
    ))

    # Optimal point (at hilly landscape minimum)
    min_loss = gd_loss(0.14, 0.14)
    fig.add_trace(go.Scatter3d(
        x=[0.14], y=[0.14], z=[min_loss],
        mode="markers",
        marker=dict(size=12, color="gold", symbol="diamond",
                    line=dict(color="black", width=1)),
        name="Optimum",
    ))

    fig.update_layout(
        title=dict(
            text=f"3D Loss Surface — {gd_type}<br>Step {idx}/{gd_steps}  |  Loss: {losses[-1]:.4f}",
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title="w₁",
            yaxis_title="w₂",
            zaxis_title="Loss",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        showlegend=False,
    )
    return fig


def _draw_gd_2d_mpl(idx):
    """Return a matplotlib 2D contour figure for frame *idx*."""
    sub = gd_path[: idx + 1]
    losses = gd_loss(sub[:, 0], sub[:, 1])

    fig, ax2d = plt.subplots(figsize=(7, 6))
    ax2d.contourf(W1g, W2g, Lg, levels=30, cmap="viridis", alpha=0.85)
    ax2d.contour(W1g, W2g, Lg, levels=15, colors="white", linewidths=0.4, alpha=0.3)
    ax2d.plot(sub[:, 0], sub[:, 1], "r.-", linewidth=1.5, markersize=3)
    ax2d.scatter(
        sub[-1, 0], sub[-1, 1],
        color="red", s=90, zorder=10, edgecolors="white", linewidth=2,
    )
    ax2d.scatter(0.14, 0.14, color="gold", s=200, marker="*", zorder=10, edgecolors="black", linewidth=1.5)
    ax2d.set_xlabel("w₁")
    ax2d.set_ylabel("w₂")
    ax2d.set_title(
        f"2D Contour — {gd_type}\nStep {idx}/{gd_steps}  |  Loss: {losses[-1]:.4f}",
        fontsize=11, fontweight="bold",
    )
    ax2d.set_xlim(-3.5, 4.5)
    ax2d.set_ylim(-3.5, 4.5)
    ax2d.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


if show_gd_animation:
    col_3d, col_2d = st.columns(2)
    with col_3d:
        gd_3d_ph = st.empty()
    with col_2d:
        gd_2d_ph = st.empty()
    gd_info_ph = st.empty()

    while True:
        for i in range(1, len(gd_path)):
            # Interactive 3D
            fig_3d = _draw_gd_3d_plotly(i)
            gd_3d_ph.plotly_chart(fig_3d, use_container_width=True, key=f"gd3d_{i}_{time.time()}")

            # 2D contour
            fig_2d = _draw_gd_2d_mpl(i)
            gd_2d_ph.image(fig_to_pil(fig_2d), width="stretch")
            plt.close(fig_2d)

            cur = gd_path[i]
            cur_loss = gd_loss(cur[0], cur[1])
            gd_info_ph.markdown(
                f"**Step {i}/{gd_steps}** — w₁ = {cur[0]:.4f},  w₂ = {cur[1]:.4f}  |  "
                f"Loss = {cur_loss:.4f}"
            )
            time.sleep(anim_delay)

        gd_info_ph.markdown("### ✅ Optimisation complete! Restarting …")
        time.sleep(anim_delay * 1.5)

else:
    # Static: show full trajectory with interactive 3D
    col_3d, col_2d = st.columns(2)
    with col_3d:
        fig_3d = _draw_gd_3d_plotly(len(gd_path) - 1)
        st.plotly_chart(fig_3d, use_container_width=True)
    with col_2d:
        fig_2d = _draw_gd_2d_mpl(len(gd_path) - 1)
        st.image(fig_to_pil(fig_2d), width="stretch")
        plt.close(fig_2d)

    final_loss = gd_loss(gd_path[-1, 0], gd_path[-1, 1])
    st.markdown(
        f"**Final** — w₁ = {gd_path[-1, 0]:.4f},  w₂ = {gd_path[-1, 1]:.4f}  |  "
        f"Loss = {final_loss:.4f}  (after {gd_steps} steps with α = {gd_lr})"
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: ADVANCED OPTIMIZERS — MOMENTUM, RMSPROP, ADAM
# ═══════════════════════════════════════════════════════════════════════════════

st.header("🚀 Step 5: Advanced Optimizers — Momentum, RMSprop & Adam")
st.markdown(
    "Compare how advanced optimizers navigate the loss surface more efficiently than vanilla SGD.  \n"
    "- **SGD + Momentum**: Accumulates velocity to accelerate through flat regions and dampen oscillations.  \n"
    "- **RMSprop**: Adapts learning rate per-parameter using running average of squared gradients.  \n"
    "- **Adam**: Combines momentum and RMSprop with bias correction for robust optimization."
)

col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
with col_opt1:
    opt_type = st.selectbox(
        "Optimizer",
        ["SGD + Momentum", "RMSprop", "Adam"],
        key="opt_type",
    )
with col_opt2:
    opt_lr = st.slider(
        "Learning Rate", 0.001, 0.5, 0.1,
        step=0.005, format="%.3f", key="opt_lr",
    )
with col_opt3:
    opt_steps = st.slider("Steps", 10, 150, 50, key="opt_steps")
with col_opt4:
    show_opt_animation = st.checkbox("Animate Optimizer", value=False, key="show_opt_anim")

# Additional hyperparameters
col_hyp1, col_hyp2, col_hyp3 = st.columns(3)
with col_hyp1:
    momentum_beta = st.slider("Momentum β (for SGD+Momentum/Adam)", 0.0, 0.99, 0.9, step=0.01, key="momentum_beta")
with col_hyp2:
    rmsprop_beta = st.slider("RMSprop β / Adam β₂", 0.9, 0.999, 0.999, step=0.001, format="%.3f", key="rmsprop_beta")
with col_hyp3:
    epsilon = st.select_slider("ε (numerical stability)", options=[1e-8, 1e-7, 1e-6, 1e-5], value=1e-8, key="epsilon")

# ── Loss surface (same hilly landscape as gradient descent) ─────────────────

def opt_loss(w1, w2):
    """Hilly landscape with spread-out bumps."""
    base = 0.3 * ((w1 - 1.0) ** 2 + (w2 - 1.0) ** 2)
    hills = 3.0 * np.sin(0.8 * w1) ** 2 + 3.0 * np.sin(0.8 * w2) ** 2
    return base + hills


def opt_gradient(w1, w2):
    dw1 = 0.6 * (w1 - 1.0) + 2.4 * np.sin(1.6 * w1)
    dw2 = 0.6 * (w2 - 1.0) + 2.4 * np.sin(1.6 * w2)
    return np.array([dw1, dw2])


def generate_optimizer_path(optimizer, lr, n_steps, beta1, beta2, eps, start=(-2.0, -2.0)):
    """Simulate optimization trajectory for advanced optimizers."""
    w = np.array(start, dtype=float)
    trajectory = [w.copy()]
    
    # State variables
    v = np.zeros(2)  # velocity (momentum)
    s = np.zeros(2)  # squared gradient cache (RMSprop/Adam)
    t = 0  # timestep for Adam bias correction
    
    for _ in range(n_steps):
        t += 1
        g = opt_gradient(w[0], w[1])
        
        if optimizer == "SGD + Momentum":
            # v = β*v + (1-β)*g, then w = w - lr*v
            v = beta1 * v + (1 - beta1) * g
            w = w - lr * v
            
        elif optimizer == "RMSprop":
            # s = β*s + (1-β)*g², then w = w - lr*g/√(s+ε)
            s = beta2 * s + (1 - beta2) * (g ** 2)
            w = w - lr * g / (np.sqrt(s) + eps)
            
        elif optimizer == "Adam":
            # Momentum: v = β1*v + (1-β1)*g
            v = beta1 * v + (1 - beta1) * g
            # RMSprop: s = β2*s + (1-β2)*g²
            s = beta2 * s + (1 - beta2) * (g ** 2)
            # Bias correction
            v_corrected = v / (1 - beta1 ** t)
            s_corrected = s / (1 - beta2 ** t)
            # Update
            w = w - lr * v_corrected / (np.sqrt(s_corrected) + eps)
        
        trajectory.append(w.copy())
    
    return np.array(trajectory)


# Pre-compute path & mesh for optimizers
opt_path = generate_optimizer_path(opt_type, opt_lr, opt_steps, momentum_beta, rmsprop_beta, epsilon)

w1_opt_grid = np.linspace(-3.5, 4.5, 80)
w2_opt_grid = np.linspace(-3.5, 4.5, 80)
W1_opt, W2_opt = np.meshgrid(w1_opt_grid, w2_opt_grid)
L_opt = opt_loss(W1_opt, W2_opt)


def _draw_opt_3d_plotly(idx):
    """Return an interactive Plotly 3D figure for optimizer visualization."""
    sub = opt_path[: idx + 1]
    losses = opt_loss(sub[:, 0], sub[:, 1])

    fig = go.Figure()

    # Loss surface
    fig.add_trace(go.Surface(
        x=W1_opt, y=W2_opt, z=L_opt,
        colorscale="Plasma", opacity=0.7,
        showscale=False,
        name="Loss Surface",
    ))

    # Trajectory path
    fig.add_trace(go.Scatter3d(
        x=sub[:, 0], y=sub[:, 1], z=losses,
        mode="lines+markers",
        marker=dict(size=4, color="cyan"),
        line=dict(color="cyan", width=4),
        name="Optimizer Path",
    ))

    # Current point
    fig.add_trace(go.Scatter3d(
        x=[sub[-1, 0]], y=[sub[-1, 1]], z=[losses[-1]],
        mode="markers",
        marker=dict(size=10, color="cyan", symbol="circle",
                    line=dict(color="white", width=2)),
        name="Current",
    ))

    # Optimal point (at hilly landscape minimum)
    min_loss = opt_loss(0.14, 0.14)
    fig.add_trace(go.Scatter3d(
        x=[0.14], y=[0.14], z=[min_loss],
        mode="markers",
        marker=dict(size=12, color="gold", symbol="diamond",
                    line=dict(color="black", width=1)),
        name="Optimum",
    ))

    fig.update_layout(
        title=dict(
            text=f"3D Loss Surface — {opt_type}<br>Step {idx}/{opt_steps}  |  Loss: {losses[-1]:.4f}",
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title="w₁",
            yaxis_title="w₂",
            zaxis_title="Loss",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        showlegend=False,
    )
    return fig


def _draw_opt_2d_mpl(idx):
    """Return a matplotlib 2D contour figure for optimizer visualization."""
    sub = opt_path[: idx + 1]
    losses = opt_loss(sub[:, 0], sub[:, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(W1_opt, W2_opt, L_opt, levels=30, cmap="plasma", alpha=0.85)
    ax.contour(W1_opt, W2_opt, L_opt, levels=15, colors="white", linewidths=0.4, alpha=0.3)
    ax.plot(sub[:, 0], sub[:, 1], "c.-", linewidth=2, markersize=4)
    ax.scatter(
        sub[-1, 0], sub[-1, 1],
        color="cyan", s=100, zorder=10, edgecolors="white", linewidth=2,
    )
    ax.scatter(0.14, 0.14, color="gold", s=200, marker="*", zorder=10, edgecolors="black", linewidth=1.5)
    ax.set_xlabel("w₁")
    ax.set_ylabel("w₂")
    ax.set_title(
        f"2D Contour — {opt_type}\nStep {idx}/{opt_steps}  |  Loss: {losses[-1]:.4f}",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(-3.5, 4.5)
    ax.set_ylim(-3.5, 4.5)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


# Display optimizer info
st.markdown("### 📝 Optimizer Equations")
if opt_type == "SGD + Momentum":
    st.latex(r"v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot \nabla L")
    st.latex(r"w_t = w_{t-1} - \alpha \cdot v_t")
    st.markdown(f"**Current:** β = {momentum_beta}, α = {opt_lr}")
elif opt_type == "RMSprop":
    st.latex(r"s_t = \beta \cdot s_{t-1} + (1 - \beta) \cdot (\nabla L)^2")
    st.latex(r"w_t = w_{t-1} - \alpha \cdot \frac{\nabla L}{\sqrt{s_t} + \epsilon}")
    st.markdown(f"**Current:** β = {rmsprop_beta}, α = {opt_lr}, ε = {epsilon}")
else:  # Adam
    st.latex(r"v_t = \beta_1 \cdot v_{t-1} + (1 - \beta_1) \cdot \nabla L \quad \text{(momentum)}")
    st.latex(r"s_t = \beta_2 \cdot s_{t-1} + (1 - \beta_2) \cdot (\nabla L)^2 \quad \text{(RMSprop)}")
    st.latex(r"\hat{v}_t = \frac{v_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t} \quad \text{(bias correction)}")
    st.latex(r"w_t = w_{t-1} - \alpha \cdot \frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \epsilon}")
    st.markdown(f"**Current:** β₁ = {momentum_beta}, β₂ = {rmsprop_beta}, α = {opt_lr}, ε = {epsilon}")


if show_opt_animation:
    col_opt_3d, col_opt_2d = st.columns(2)
    with col_opt_3d:
        opt_3d_ph = st.empty()
    with col_opt_2d:
        opt_2d_ph = st.empty()
    opt_info_ph = st.empty()

    while True:
        for i in range(1, len(opt_path)):
            # Interactive 3D
            fig_3d = _draw_opt_3d_plotly(i)
            opt_3d_ph.plotly_chart(fig_3d, use_container_width=True, key=f"opt3d_{i}_{time.time()}")

            # 2D contour
            fig_2d = _draw_opt_2d_mpl(i)
            opt_2d_ph.image(fig_to_pil(fig_2d), width="stretch")
            plt.close(fig_2d)

            cur = opt_path[i]
            cur_loss = opt_loss(cur[0], cur[1])
            opt_info_ph.markdown(
                f"**Step {i}/{opt_steps}** — w₁ = {cur[0]:.4f},  w₂ = {cur[1]:.4f}  |  "
                f"Loss = {cur_loss:.4f}"
            )
            time.sleep(anim_delay)

        opt_info_ph.markdown("### ✅ Optimization complete! Restarting …")
        time.sleep(anim_delay * 1.5)

else:
    # Static: show full trajectory with interactive 3D
    col_opt_3d, col_opt_2d = st.columns(2)
    with col_opt_3d:
        fig_3d = _draw_opt_3d_plotly(len(opt_path) - 1)
        st.plotly_chart(fig_3d, use_container_width=True, key="opt_3d_static")
    with col_opt_2d:
        fig_2d = _draw_opt_2d_mpl(len(opt_path) - 1)
        st.image(fig_to_pil(fig_2d), width="stretch")
        plt.close(fig_2d)

    final_opt_loss = opt_loss(opt_path[-1, 0], opt_path[-1, 1])
    st.markdown(
        f"**Final** — w₁ = {opt_path[-1, 0]:.4f},  w₂ = {opt_path[-1, 1]:.4f}  |  "
        f"Loss = {final_opt_loss:.4f}  (after {opt_steps} steps)"
    )
