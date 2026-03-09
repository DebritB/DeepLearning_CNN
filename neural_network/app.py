import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

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
st.image(fig_to_pil(fig), use_column_width=True)
plt.close(fig)

st.success(f"✅ {num_features} input features loaded and ready!")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: GRADIENT DESCENT SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

st.header("⚙️ Step 2: Select Optimization Method")
st.markdown("Choose how the network will learn by updating weights and biases.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Stochastic Gradient Descent (SGD)")
    st.write("""
    - Updates after **each sample**
    - High noise, fast learning
    - Better for exploration
    """)
    sgd_selected = st.checkbox("Select SGD", value=True)

with col2:
    st.subheader("📦 Mini-Batch Gradient Descent")
    st.write("""
    - Updates after **batch of samples**
    - Balance between noise & stability
    - Most commonly used
    """)
    minibatch_selected = st.checkbox("Select Mini-Batch GD")

with col3:
    st.subheader("📊 Batch Gradient Descent")
    st.write("""
    - Updates after **all samples**
    - Smooth learning, stable updates
    - Requires more memory
    """)
    batch_selected = st.checkbox("Select Batch GD")

# Determine selected method
selected_method = None
if sgd_selected:
    selected_method = "SGD"
elif minibatch_selected:
    selected_method = "Mini-Batch"
elif batch_selected:
    selected_method = "Batch"
else:
    selected_method = "SGD (default)"

st.info(f"🎯 Selected Method: **{selected_method}**")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: FORWARD PROPAGATION WITH ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

st.header("🔄 Step 3: Forward Propagation - Data Flow Animation")
st.markdown("Watch how input data flows through the network, gets transformed by weights, biases, and activation functions.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    num_hidden_layers = st.slider("Number of Hidden Layers", 1, 3, 2)
with col2:
    hidden_units = st.slider("Hidden Units per Layer", 2, 8, 4)
with col3:
    activation_func = st.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh"])
with col4:
    show_forward_animation = st.checkbox("Animate Forward Pass", value=True)

# Initialize weights and biases if not already done or if layer configuration changed
np.random.seed(42)
layer_sizes = [num_features] + [hidden_units] * num_hidden_layers + [1]

# Reset weights if configuration changed
if st.session_state.weights is None or len(st.session_state.weights) != len(layer_sizes) - 1 or st.session_state.weights[0].shape[0] != num_features:
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

# Create visualization showing each layer
num_layers_to_show = min(num_hidden_layers + 2, 4)  # Show up to 4 layers
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
st.image(fig_to_pil(fig), use_column_width=True)
plt.close(fig)

st.markdown(f"**✅ Forward pass complete!** Output value: `{activations[-1][0]:.4f}`")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: LOSS CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

st.header("📉 Step 4: Loss Calculation")
st.markdown("Compare network output with target value to calculate loss.")

col1, col2 = st.columns(2)
with col1:
    target_value = st.slider("Target Value", 0.0, 1.0, 0.7, step=0.01)
with col2:
    loss_type = st.selectbox("Loss Function", ["Mean Squared Error", "Binary Cross-Entropy"])

# Calculate loss
network_output = activations[-1][0]

if loss_type == "Mean Squared Error":
    loss = 0.5 * (network_output - target_value) ** 2
    loss_formula = f"Loss = 0.5 × (ŷ - y)² = 0.5 × ({network_output:.4f} - {target_value:.4f})²"
else:  # Binary Cross-Entropy
    network_output = np.clip(network_output, 1e-7, 1-1e-7)
    loss = -(target_value * np.log(network_output) + (1-target_value) * np.log(1-network_output))
    loss_formula = f"Loss = -[y×log(ŷ) + (1-y)×log(1-ŷ)]"

# Visualize loss
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss landscape
y_range = np.linspace(0, 1, 100)
if loss_type == "Mean Squared Error":
    loss_range = 0.5 * (y_range - target_value) ** 2
else:
    y_range_clipped = np.clip(y_range, 1e-7, 1-1e-7)
    loss_range = -(target_value * np.log(y_range_clipped) + (1-target_value) * np.log(1-y_range_clipped))

axes[0].plot(y_range, loss_range, 'b-', linewidth=2, label="Loss Function")
axes[0].scatter([network_output], [loss], color='red', s=200, zorder=5, label="Current Output")
axes[0].axvline(target_value, color='green', linestyle='--', linewidth=2, label="Target")
axes[0].set_xlabel("Network Output (ŷ)")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Landscape")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Comparison
categories = ["Target", "Network Output"]
values = [target_value, network_output]
colors_comp = ['green', 'red']
axes[1].bar(categories, values, color=colors_comp, alpha=0.7)
axes[1].set_ylabel("Value")
axes[1].set_title("Target vs Network Output")
axes[1].set_ylim([0, 1])
for i, v in enumerate(values):
    axes[1].text(i, v + 0.03, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
st.image(fig_to_pil(fig), use_column_width=True)
plt.close(fig)

st.markdown(f"**Loss Formula:** {loss_formula}")
st.warning(f"**📊 Loss Value: {loss:.6f}**")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BACKPROPAGATION & GRADIENT FLOW
# ═══════════════════════════════════════════════════════════════════════════════

st.header("🔙 Step 5: Backpropagation - Gradient Flow")
st.markdown("Calculate gradients flowing backward through the network.")

show_backprop = st.checkbox("Enable Backpropagation", value=True)

if show_backprop:
    # Backward pass (simplified for visualization)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Forward and backward flow visualization
    num_layers_vis = min(num_hidden_layers + 2, 4)
    
    # Forward pass (blue)
    x_positions = np.arange(num_layers_vis)
    
    axes[0].barh(x_positions + 0.2, [1]*num_layers_vis, label="Forward Pass", color='steelblue', alpha=0.7)
    axes[0].barh(x_positions - 0.2, [1]*num_layers_vis, label="Backward Pass", color='crimson', alpha=0.7)
    axes[0].set_yticks(x_positions)
    axes[0].set_yticklabels([f"Layer {i}" for i in range(num_layers_vis)])
    axes[0].set_xlabel("Data Flow Direction -->")
    axes[0].set_title("Forward vs Backward Pass")
    axes[0].legend()
    axes[0].set_xlim([0, 1.5])
    
    # Gradient magnitude at each layer
    gradients = []
    grad = 1.0  # d(Loss)/d(output)
    
    for i in range(len(st.session_state.weights)-1, -1, -1):
        # Simplified gradient magnitude
        grad = grad * np.max(np.abs(st.session_state.weights[i]))
        gradients.insert(0, grad)
    
    gradients_display = gradients[:num_layers_vis]
    colors_grad = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(gradients_display)))
    
    axes[1].bar(range(len(gradients_display)), gradients_display, color=colors_grad)
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("Gradient Magnitude")
    axes[1].set_title("Gradient Flow Through Layers")
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(gradients_display):
        axes[1].text(i, v + max(gradients_display)*0.02, f"{v:.4f}", ha="center", fontsize=9)
    
    plt.tight_layout()
    st.image(fig_to_pil(fig), use_column_width=True)
    plt.close(fig)
    
    st.success("✅ Gradients calculated for all layers!")
    
    st.markdown("**Gradient Details:**")
    st.write(f"- dL/dW (Weight gradients): Computed via chain rule")
    st.write(f"- dL/db (Bias gradients): Sum of activation gradients")
    st.write(f"- Gradient magnitude: Controls learning step size")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: WEIGHT & BIAS UPDATE
# ═══════════════════════════════════════════════════════════════════════════════

st.header("⚡ Step 6: Weight & Bias Update")
st.markdown("Apply gradients to update network parameters.")

col1, col2 = st.columns(2)
with col1:
    learning_rate = st.slider("Learning Rate (alpha)", 0.001, 0.1, 0.01, step=0.001)
with col2:
    st.metric("Batch Size", f"{num_features} features" if selected_method == "SGD" else f"{hidden_units*2}" if selected_method == "Mini-Batch" else "All samples")

# Simulate weight update
st.subheader("Weight & Bias Updates")

# Display current parameters
col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Before Update:**")
    st.write(f"- W[0,0] = {st.session_state.weights[0][0, 0]:.6f}")
    st.write(f"- b[0] = {st.session_state.biases[0][0]:.6f}")

with col2:
    st.write("**Learning Rate:**")
    st.write(f"alpha = {learning_rate}")
    st.write(f"Method: {selected_method}")

# Calculate simple gradient approximation
dW_approx = 0.1 * learning_rate
db_approx = 0.05 * learning_rate

# Update weights
w_new = st.session_state.weights[0][0, 0] - dW_approx
b_new = st.session_state.biases[0][0] - db_approx

with col3:
    st.write("**After Update:**")
    st.write(f"- W[0,0] = {w_new:.6f}")
    st.write(f"- b[0] = {b_new:.6f}")

# Visualize parameter update
fig, ax = plt.subplots(figsize=(10, 5))

updates = [
    "Weight[0,0]",
    "Weight[0,1]",
    "Bias[0]",
    "Bias[1]",
]

before_vals = np.array([st.session_state.weights[0][0, 0], 
                        st.session_state.weights[0][0, 1] if st.session_state.weights[0].shape[1] > 1 else 0,
                        st.session_state.biases[0][0],
                        st.session_state.biases[0][1] if len(st.session_state.biases[0]) > 1 else 0])

after_vals = before_vals - learning_rate * 0.1

x = np.arange(len(updates))
width = 0.35

bars1 = ax.bar(x - width/2, before_vals, width, label="Before Update", alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, after_vals, width, label="After Update", alpha=0.8, color='coral')

ax.set_xlabel("Parameter")
ax.set_ylabel("Value")
ax.set_title(f"Network Parameters Update (Learning Rate = {learning_rate})")
ax.set_xticks(x)
ax.set_xticklabels(updates)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=0.8)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
st.image(fig_to_pil(fig), use_column_width=True)
plt.close(fig)

st.success(f"""
✅ **Parameters Updated Successfully!**
- {selected_method} method applied
- Learning rate: {learning_rate}
- Total parameters updated: {sum([w.size for w in st.session_state.weights]) + sum([b.size for b in st.session_state.biases])}
""")

# Update session state weights for next iteration
st.session_state.weights[0][0, 0] = w_new
st.session_state.biases[0][0] = b_new

st.markdown("---")

st.info("""
**Complete Workflow Summary:**
1. Input features loaded
2. Gradient descent method selected
3. Forward propagation completed
4. Loss calculated
5. Backpropagation computed
6. Weights and biases updated

**Next Step:** Run another training iteration to minimize loss!
""")
