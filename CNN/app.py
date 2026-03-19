import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image
import io
import time
from typing import Tuple, List

# Set dark background style
plt.style.use('dark_background')

# Helper function to convert matplotlib figure to PIL image
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor='#0a0a0a')
    buf.seek(0)
    return Image.open(buf)

st.set_page_config(page_title="CNN Convolution Visualizer", layout="wide")
st.markdown("""
    <style>
        body { background-color: #0a0a0a; color: #ffffff; }
        .stApp { background-color: #0a0a0a; }
        h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
        p, span, div { color: #ffffff !important; }
        [data-baseweb="select"] { background-color: #1a1a1a !important; }
        [data-baseweb="select"] span { color: #ffffff !important; }
        [data-baseweb="popover"] { background-color: #1a1a1a !important; }
        .stSelectbox > div > div { background-color: #1a1a1a !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 CNN: Convolution Operation Visualizer")

st.markdown("""
This interactive tool visualizes how **convolutional operations** work in CNNs.
Watch the kernel **slide across the input** with a glossy 3D effect.
Configure **padding** and **stride** parameters to see how they affect the output.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONVOLUTION SETUP
# ═══════════════════════════════════════════════════════════════════════════════

st.header("⚙️ Convolution Configuration")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    input_size = st.slider("Input Size (H×W)", 5, 15, 8)

with col2:
    kernel_size = st.slider("Kernel Size", 2, 5, 3)

with col3:
    padding = st.slider("Padding", 0, 3, 1)

with col4:
    stride = st.slider("Stride", 1, 3, 1)

with col5:
    anim_speed = st.slider("Animation Speed", 1, 10, 5, help="Higher = faster")

anim_delay = 2.0 / anim_speed

# Calculate output dimensions
def calculate_output_size(input_size, kernel_size, padding, stride):
    """Calculate output feature map size."""
    return ((input_size + 2 * padding - kernel_size) // stride) + 1

output_size = calculate_output_size(input_size, kernel_size, padding, stride)

# Generate random input and kernel
np.random.seed(42)
input_map = np.random.rand(input_size, input_size)
kernel = np.random.randn(kernel_size, kernel_size) * 0.5

# Pad the input
padded_input = np.pad(input_map, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONVOLUTION MATH & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

st.header("📊 Convolution Process")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.subheader("📐 Configuration Details")
    details = rf"""
    **Input Size:** {input_size} × {input_size}
    
    **Kernel Size:** {kernel_size} × {kernel_size}
    
    **Padding:** {padding}
    
    **Padded Input Size:** {padded_input.shape[0]} × {padded_input.shape[1]}
    
    **Stride:** {stride}
    
    **Output Size:** {output_size} × {output_size}
    
    **Formula:** 
    $$O = \left\lfloor \frac{{I + 2P - K}}{{S}} \right\rfloor + 1$$
    
    Where:
    - I = Input size
    - K = Kernel size
    - P = Padding
    - S = Stride
    """
    st.markdown(details)

with col_info2:
    st.subheader("🔢 Kernel Matrix (Random Filter)")
    fig_kernel = plt.figure(figsize=(6, 5), facecolor='#0a0a0a')
    ax = fig_kernel.add_subplot(111)
    ax.set_facecolor('#1a1a1a')
    im = ax.imshow(kernel, cmap='hot', interpolation='nearest', alpha=0.9)
    ax.set_title(f"Kernel ({kernel_size}×{kernel_size})", fontweight='bold', fontsize=12, color='#ffffff', pad=15)
    for i in range(kernel_size):
        for j in range(kernel_size):
            val = kernel[i, j]
            color = '#ffffff' if abs(val) > abs(kernel).max() / 2 else '#cccccc'
            ax.text(j, i, f'{kernel[i, j]:.2f}', ha='center', va='center',
                   color=color, fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#000000', alpha=0.6, edgecolor='#ff6600'))
    ax.set_xticks(range(kernel_size))
    ax.set_yticks(range(kernel_size))
    ax.grid(True, alpha=0.2, color='#ff6600')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors='#cccccc')
    fig_kernel.patch.set_facecolor('#0a0a0a')
    plt.tight_layout()
    st.image(fig_to_pil(fig_kernel), width='stretch')
    plt.close(fig_kernel)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: KERNEL FLYING OVER INPUT WITH SHADOW
# ═══════════════════════════════════════════════════════════════════════════════

st.header("🎬 Kernel Flying Over Input with Shadow")

show_animation = st.checkbox("Animate Convolution Process", value=False)

def draw_kernel_shadow_conv(padded_input, kernel, kernel_size, h, w, output_map, out_h, out_w, step_num, total_steps):
    """Draw input, kernel, and output side by side with connecting lines."""
    fig = plt.figure(figsize=(14, 6), facecolor='#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0a0a0a')
    
    h_in, w_in = padded_input.shape
    cell_size = 1.0
    y_center = 0
    spacing = 2.0
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Draw INPUT matrix on the LEFT
    # ═════════════════════════════════════════════════════════════════════════════
    input_x_offset = 0
    
    for i in range(h_in):
        for j in range(w_in):
            x = input_x_offset + j * cell_size
            y = y_center - i * cell_size
            
            # Check if this cell is in receptive field (shadow)
            in_shadow = (h <= i < h + kernel_size) and (w <= j < w + kernel_size)
            
            # Draw cell background
            if in_shadow:
                rect_color = '#ff6600'
                alpha = 0.3
                edge_color = '#ff6600'
                edge_width = 2
            else:
                rect_color = '#00aaff'
                alpha = 0.1
                edge_color = '#00aaff'
                edge_width = 0.5
            
            rect = Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, 
                           facecolor=rect_color, alpha=alpha, 
                           edgecolor=edge_color, linewidth=edge_width)
            ax.add_patch(rect)
            
            # Draw number
            text_color = '#ff6600' if in_shadow else '#cccccc'
            text_weight = 'bold' if in_shadow else 'normal'
            ax.text(x, y, f'{padded_input[i, j]:.1f}', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight=text_weight)
    
    # Shadow box on input
    shadow_x = [input_x_offset + w * cell_size - 0.5, 
                input_x_offset + (w + kernel_size) * cell_size - 0.5, 
                input_x_offset + (w + kernel_size) * cell_size - 0.5, 
                input_x_offset + w * cell_size - 0.5, 
                input_x_offset + w * cell_size - 0.5]
    shadow_y = [y_center - (h - 1) * cell_size - 0.5, 
                y_center - (h - 1) * cell_size - 0.5,
                y_center - (h + kernel_size - 1) * cell_size - 0.5, 
                y_center - (h + kernel_size - 1) * cell_size - 0.5,
                y_center - (h - 1) * cell_size - 0.5]
    ax.plot(shadow_x, shadow_y, color='#ff0000', linewidth=2.5, linestyle='--', alpha=0.8)
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Draw KERNEL in the MIDDLE
    # ═════════════════════════════════════════════════════════════════════════════
    kernel_x_offset = input_x_offset + w_in * cell_size + spacing
    kernel_y_center = y_center - (h_in - kernel_size) * cell_size / 2.0  # Center vertically
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = kernel_x_offset + j * cell_size
            y = kernel_y_center - i * cell_size
            
            # Draw cell background
            rect = Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                           facecolor='#ff6600', alpha=0.5,
                           edgecolor='#ffff00', linewidth=1.5)
            ax.add_patch(rect)
            
            # Draw number
            val = kernel[i, j]
            text_color = '#ffff00' if abs(val) > abs(kernel).max() / 2 else '#ffffff'
            ax.text(x, y, f'{val:.2f}', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')
    
    # Perspective lines connecting kernel to shadow
    kernel_corners = [
        (kernel_x_offset - 0.5, kernel_y_center + 0.5),  # Top-left
        (kernel_x_offset + kernel_size * cell_size - 0.5, kernel_y_center + 0.5),  # Top-right
        (kernel_x_offset + kernel_size * cell_size - 0.5, kernel_y_center - kernel_size * cell_size + 0.5),  # Bottom-right
        (kernel_x_offset - 0.5, kernel_y_center - kernel_size * cell_size + 0.5),  # Bottom-left
    ]
    
    shadow_corners = [
        (input_x_offset + w * cell_size - 0.5, y_center - (h - 1) * cell_size - 0.5),  # Top-left
        (input_x_offset + (w + kernel_size) * cell_size - 0.5, y_center - (h - 1) * cell_size - 0.5),  # Top-right
        (input_x_offset + (w + kernel_size) * cell_size - 0.5, y_center - (h + kernel_size - 1) * cell_size - 0.5),  # Bottom-right
        (input_x_offset + w * cell_size - 0.5, y_center - (h + kernel_size - 1) * cell_size - 0.5),  # Bottom-left
    ]
    
    for kernel_corner, shadow_corner in zip(kernel_corners, shadow_corners):
        ax.plot([kernel_corner[0], shadow_corner[0]], [kernel_corner[1], shadow_corner[1]],
               color='#ff0000', linewidth=2, linestyle='--', alpha=0.7)
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Draw OUTPUT on the RIGHT
    # ═════════════════════════════════════════════════════════════════════════════
    out_h_size, out_w_size = output_map.shape
    output_x_offset = kernel_x_offset + kernel_size * cell_size + spacing
    output_y_center = y_center - (h_in - out_h_size) * cell_size / 2.0  # Center vertically
    
    for i in range(out_h_size):
        for j in range(out_w_size):
            x = output_x_offset + j * cell_size
            y = output_y_center - i * cell_size
            
            is_current = (i == out_h and j == out_w)
            val = output_map[i, j]
            
            # Draw cell
            rect = Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                           facecolor='#ff00ff', alpha=0.4 if is_current else 0.2,
                           edgecolor='#ff00ff' if is_current else '#666666', 
                           linewidth=2 if is_current else 0.5)
            ax.add_patch(rect)
            
            # Draw number
            color = '#ffff00' if is_current else '#cccccc'
            size = 9
            ax.text(x, y, f'{val:.1f}', ha='center', va='center',
                   color=color, fontsize=size, fontweight='bold')
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Set axis properties
    # ═════════════════════════════════════════════════════════════════════════════
    ax.set_xlim(-1, output_x_offset + out_w_size * cell_size + 1)
    ax.set_ylim(-h_in * cell_size - 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    title = f'Convolution Animation — Step {step_num}/{total_steps}\n'
    title += f'Kernel at ({h}, {w}) | Output[{out_h}, {out_w}] = {output_map[out_h, out_w]:.2f}'
    ax.text(0.5, 1.05, title, transform=ax.transAxes, ha='center', va='bottom',
           fontsize=12, fontweight='bold', color='#ffffff')
    
    plt.tight_layout()
    return fig

if show_animation:
    anim_placeholder = st.empty()
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        info_placeholder = st.empty()
    
    with col_info2:
        math_placeholder = st.empty()
    
    # Calculate all positions
    positions = []
    h_pos = 0
    while h_pos + kernel_size <= padded_input.shape[0]:
        w_pos = 0
        while w_pos + kernel_size <= padded_input.shape[1]:
            positions.append((h_pos, w_pos, h_pos // stride, w_pos // stride))
            w_pos += stride
        h_pos += stride
    
    while True:  # Loop animation
        output_map = np.zeros((output_size, output_size))
        
        for idx, (h, w, out_h, out_w) in enumerate(positions):
            # Extract receptive field
            receptive_field = padded_input[h:h+kernel_size, w:w+kernel_size]
            
            # Compute convolution
            conv_result = np.sum(receptive_field * kernel)
            output_map[out_h, out_w] = conv_result
            
            # Draw visualization
            fig = draw_kernel_shadow_conv(padded_input, kernel, kernel_size, h, w, 
                                         output_map, out_h, out_w, idx + 1, len(positions))
            anim_placeholder.image(fig_to_pil(fig), width='stretch')
            plt.close(fig)
            
            # Show computation details
            info_text = f"""
### 📍 Step {idx + 1} / {len(positions)}

**Kernel Position:** ({h}, {w})  
**Output Position:** [{out_h}, {out_w}]  
**Output Value:** `{conv_result:.4f}`

---

**Receptive Field (Shadow on Input):**
"""
            for i, row in enumerate(receptive_field):
                info_text += "\n`["
                for j, val in enumerate(row):
                    info_text += f"{val:7.2f}"
                info_text += "]`"
            
            info_placeholder.markdown(info_text)
            
            # Math details
            math_text = f"""
**Kernel (Floating Filter):**
"""
            for i, row in enumerate(kernel):
                math_text += "\n`["
                for j, val in enumerate(row):
                    math_text += f"{val:+7.2f}"
                math_text += "]`"
            
            math_text += f"\n\n**Element-wise Product:**\n"
            result = receptive_field * kernel
            for i, row in enumerate(result):
                math_text += "`["
                for j, val in enumerate(row):
                    math_text += f"{val:+7.3f}"
                math_text += "]`\n"
            
            math_text += f"\n**Sum = {conv_result:.4f}**"
            
            math_placeholder.markdown(math_text)
            
            time.sleep(anim_delay)
        
        info_placeholder.success(f"✅ **Convolution Complete!**\n\nOutput: {output_size}×{output_size}\n\nRestarting...")
        time.sleep(2)

else:
    st.info("☑️ Check **Animate Convolution Process** to watch the kernel fly over the input.")

# Compute full convolution output for use in pooling
output_map = np.zeros((output_size, output_size))
h_pos = 0
while h_pos + kernel_size <= padded_input.shape[0]:
    w_pos = 0
    while w_pos + kernel_size <= padded_input.shape[1]:
        receptive_field = padded_input[h_pos:h_pos+kernel_size, w_pos:w_pos+kernel_size]
        conv_result = np.sum(receptive_field * kernel)
        out_h = h_pos // stride
        out_w = w_pos // stride
        output_map[out_h, out_w] = conv_result
        w_pos += stride
    h_pos += stride

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: POOLING OPERATIONS (Average & Max Pool)
# ═══════════════════════════════════════════════════════════════════════════════

st.header("🏊 Pooling Operations: Average Pool & Max Pool")

st.markdown("""
**Pooling** reduces spatial dimensions by aggregating information in local regions.
- **Max Pool:** Takes the maximum value in each pool window
- **Average Pool:** Takes the average of values in each pool window
""")

col_pool1, col_pool2, col_pool3 = st.columns(3)

with col_pool1:
    pool_type = st.selectbox("Pool Type", ["Max Pool", "Average Pool"])

with col_pool2:
    pool_size = st.slider("Pool Size", 2, 5, 2)

with col_pool3:
    pool_stride = st.slider("Pool Stride", 1, 3, 2)

show_pool_animation = st.checkbox("Animate Pooling Process", value=False)

def draw_pooling_animation(conv_output, pool_size, pool_stride, pool_type, h, w, pool_output, out_h, out_w, step_num, total_steps):
    """Draw pooling operation visualization."""
    fig = plt.figure(figsize=(20, 20), facecolor='#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0a0a0a')
    
    h_in, w_in = conv_output.shape
    cell_size = 1.0
    y_center = 0
    spacing = 2.0
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Draw INPUT (Convolution Output) on the LEFT
    # ═════════════════════════════════════════════════════════════════════════════
    input_x_offset = 0
    
    for i in range(h_in):
        for j in range(w_in):
            x = input_x_offset + j * cell_size
            y = y_center - i * cell_size
            
            # Check if in pool window
            in_pool = (h <= i < h + pool_size) and (w <= j < w + pool_size)
            
            rect_color = '#00ff00' if in_pool else '#00aaff'
            alpha = 0.35 if in_pool else 0.1
            edge_color = '#00ff00' if in_pool else '#00aaff'
            edge_width = 2 if in_pool else 0.5
            
            from matplotlib.patches import Rectangle
            rect = Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                           facecolor=rect_color, alpha=alpha,
                           edgecolor=edge_color, linewidth=edge_width)
            ax.add_patch(rect)
            
            text_color = '#00ff00' if in_pool else '#cccccc'
            text_weight = 'bold' if in_pool else 'normal'
            ax.text(x, y, f'{conv_output[i, j]:.1f}', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight=text_weight)
    
    # Pool window box (one row up)
    pool_x = [input_x_offset + w * cell_size - 0.5,
              input_x_offset + (w + pool_size) * cell_size - 0.5,
              input_x_offset + (w + pool_size) * cell_size - 0.5,
              input_x_offset + w * cell_size - 0.5,
              input_x_offset + w * cell_size - 0.5]
    pool_y = [y_center - (h - 1) * cell_size - 0.5,
              y_center - (h - 1) * cell_size - 0.5,
              y_center - (h + pool_size - 1) * cell_size - 0.5,
              y_center - (h + pool_size - 1) * cell_size - 0.5,
              y_center - (h - 1) * cell_size - 0.5]
    ax.plot(pool_x, pool_y, color='#ff0000', linewidth=2.5, linestyle='--', alpha=0.8)
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Draw FULL OUTPUT MATRIX on the RIGHT
    # ═════════════════════════════════════════════════════════════════════════════
    out_h_size, out_w_size = pool_output.shape
    output_x_offset = input_x_offset + w_in * cell_size + spacing
    output_y_center = y_center - (h_in - out_h_size) * cell_size / 2.0
    
    for i in range(out_h_size):
        for j in range(out_w_size):
            x = output_x_offset + j * cell_size
            y = output_y_center - i * cell_size
            
            is_current = (i == out_h and j == out_w)
            
            rect = Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                           facecolor='#ff00ff', alpha=0.4 if is_current else 0.2,
                           edgecolor='#ff00ff' if is_current else '#666666',
                           linewidth=2 if is_current else 0.5)
            ax.add_patch(rect)
            
            color = '#ffff00' if is_current else '#cccccc'
            ax.text(x, y, f'{pool_output[i, j]:.1f}', ha='center', va='center',
                   color=color, fontsize=9, fontweight='bold')
    
    # Get corners of the current output cell
    output_cell_x = output_x_offset + out_w * cell_size
    output_cell_y = output_y_center - out_h * cell_size
    
    output_corners = [
        (output_cell_x - 0.45, output_cell_y + 0.45),  # Top-left
        (output_cell_x + 0.45, output_cell_y + 0.45),  # Top-right
        (output_cell_x + 0.45, output_cell_y - 0.45),  # Bottom-right
        (output_cell_x - 0.45, output_cell_y - 0.45),  # Bottom-left
    ]
    
    # Pool boundary corners (one row up)
    pool_corners = [
        (input_x_offset + w * cell_size - 0.5, y_center - (h - 1) * cell_size - 0.5),  # Top-left
        (input_x_offset + (w + pool_size) * cell_size - 0.5, y_center - (h - 1) * cell_size - 0.5),  # Top-right
        (input_x_offset + (w + pool_size) * cell_size - 0.5, y_center - (h + pool_size - 1) * cell_size - 0.5),  # Bottom-right
        (input_x_offset + w * cell_size - 0.5, y_center - (h + pool_size - 1) * cell_size - 0.5),  # Bottom-left
    ]
    
    # Draw perspective lines from pool corners to output cell corners
    for pool_corner, output_corner in zip(pool_corners, output_corners):
        ax.plot([pool_corner[0], output_corner[0]], [pool_corner[1], output_corner[1]],
               color='#ff0000', linewidth=2, linestyle='--', alpha=0.7)
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Set axis properties
    # ═════════════════════════════════════════════════════════════════════════════
    ax.set_xlim(-1, output_x_offset + out_w_size * cell_size + 1)
    ax.set_ylim(-h_in * cell_size - 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    title = f'{pool_type} ({pool_size}×{pool_size}, stride {pool_stride}) — Step {step_num}/{total_steps}\n'
    title += f'Pool at ({h}, {w}) → Output[{out_h}, {out_w}] = {pool_output[out_h, out_w]:.2f}'
    ax.text(0.5, 1.05, title, transform=ax.transAxes, ha='center', va='bottom',
           fontsize=12, fontweight='bold', color='#ffffff')
    
    plt.tight_layout()
    return fig

if show_pool_animation:
    # Calculate output size for pooling
    pool_output_size = ((output_size - pool_size) // pool_stride) + 1
    
    anim_placeholder_pool = st.empty()
    col_info1_pool, col_info2_pool = st.columns(2)
    
    with col_info1_pool:
        info_placeholder_pool = st.empty()
    
    with col_info2_pool:
        math_placeholder_pool = st.empty()
    
    # Calculate pooling positions
    pool_positions = []
    h_pos = 0
    while h_pos + pool_size <= output_size:
        w_pos = 0
        while w_pos + pool_size <= output_size:
            pool_positions.append((h_pos, w_pos, h_pos // pool_stride, w_pos // pool_stride))
            w_pos += pool_stride
        h_pos += pool_stride
    
    while True:
        pool_output = np.zeros((pool_output_size, pool_output_size))
        
        for idx, (h, w, out_h, out_w) in enumerate(pool_positions):
            # Extract pool window
            pool_window = output_map[h:h+pool_size, w:w+pool_size]
            
            # Compute pooling
            if pool_type == "Max Pool":
                pool_result = np.max(pool_window)
            else:  # Average Pool
                pool_result = np.mean(pool_window)
            
            pool_output[out_h, out_w] = pool_result
            
            # Draw visualization
            fig = draw_pooling_animation(output_map, pool_size, pool_stride, pool_type, h, w,
                                        pool_output, out_h, out_w, idx + 1, len(pool_positions))
            anim_placeholder_pool.image(fig_to_pil(fig), width='stretch')
            plt.close(fig)
            
            # Show details
            info_text = f"""
### 📍 Pool Step {idx + 1} / {len(pool_positions)}

**Pool Position:** ({h}, {w})  
**Output Position:** [{out_h}, {out_w}]  
**Pool Result:** `{pool_result:.4f}`

---

**Pool Window Values:**
"""
            for i, row in enumerate(pool_window):
                info_text += "\n`["
                for j, val in enumerate(row):
                    info_text += f"{val:7.2f}"
                info_text += "]`"
            
            info_placeholder_pool.markdown(info_text)
            
            # Math details
            if pool_type == "Max Pool":
                math_text = f"**Operation:** Max Pool\n\n**Max Value:** `{pool_result:.4f}`"
            else:
                math_text = f"**Operation:** Average Pool\n\n**Sum:** `{np.sum(pool_window):.4f}`\n\n**Average:** `{pool_result:.4f}`"
            
            math_placeholder_pool.markdown(math_text)
            
            time.sleep(anim_delay)
        
        info_placeholder_pool.success(f"✅ **Pooling Complete!**\n\nOutput: {pool_output_size}×{pool_output_size}\n\nRestarting...")
        time.sleep(2)

else:
    st.info("☑️ Check **Animate Pooling Process** to watch the pooling operation.")

# Calculate and compute full pooling output for flatten layer
pool_output_size = ((output_size - pool_size) // pool_stride) + 1
pool_output = np.zeros((pool_output_size, pool_output_size))

h_pos = 0
while h_pos + pool_size <= output_size:
    w_pos = 0
    while w_pos + pool_size <= output_size:
        pool_window = output_map[h_pos:h_pos+pool_size, w_pos:w_pos+pool_size]
        if pool_type == "Max Pool":
            pool_result = np.max(pool_window)
        else:  # Average Pool
            pool_result = np.mean(pool_window)
        
        out_h = h_pos // pool_stride
        out_w = w_pos // pool_stride
        pool_output[out_h, out_w] = pool_result
        w_pos += pool_stride
    h_pos += pool_stride

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: FLATTEN LAYER ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

st.header("🔄 Flatten Layer: 2D to 1D Transformation")

st.markdown("""
**Flatten Layer** reshapes the 2D feature map into a 1D vector.
This is a crucial step before feeding data into **fully connected layers**.

The flattening process:
- Takes each element from the 2D matrix row by row
- Chains them into a single 1D vector
- Preserves all information while changing the shape
""")

st.subheader("📊 Flatten Configuration")

col_flat1, col_flat2 = st.columns(2)

with col_flat1:
    st.metric("Input Shape (Pooling Output)", f"{pool_output_size}×{pool_output_size}")

with col_flat2:
    st.metric("Flattened Vector Length", f"{pool_output_size * pool_output_size}")

show_flatten_animation = st.checkbox("Animate Flatten Process", value=False)

def draw_flatten_animation(pool_output, flattened_vector, current_idx, total_elements, step_num, animation_speed):
    """Draw flatten layer animation with 2D to 1D transformation."""
    fig = plt.figure(figsize=(20, 20), facecolor='#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0a0a0a')
    
    h_in, w_in = pool_output.shape
    cell_size = 0.8
    y_center = 0
    spacing = 3.0
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Draw 2D MATRIX (LEFT SIDE)
    # ═════════════════════════════════════════════════════════════════════════════
    matrix_x_offset = 0
    
    # Draw highlight path showing flattening order
    visited_positions = []
    for idx in range(min(current_idx, total_elements)):
        i = idx // w_in
        j = idx % w_in
        visited_positions.append((i, j))
    
    # Draw cells
    for i in range(h_in):
        for j in range(w_in):
            x = matrix_x_offset + j * cell_size
            y = y_center - i * cell_size
            
            # Determine cell state
            global_idx = i * w_in + j
            is_current = (global_idx == current_idx)
            is_visited = (global_idx < current_idx)
            
            # Color based on state
            if is_current:
                rect_color = '#ffff00'
                alpha = 0.7
                edge_color = '#ffffff'
                edge_width = 3
            elif is_visited:
                rect_color = '#00ff00'
                alpha = 0.5
                edge_color = '#00ff00'
                edge_width = 2
            else:
                rect_color = '#0088ff'
                alpha = 0.15
                edge_color = '#0088ff'
                edge_width = 0.5
            
            rect = Rectangle((x - cell_size/2 + 0.05, y - cell_size/2 + 0.05), 
                           cell_size - 0.1, cell_size - 0.1,
                           facecolor=rect_color, alpha=alpha,
                           edgecolor=edge_color, linewidth=edge_width)
            ax.add_patch(rect)
            
            # Draw value
            text_color = '#000000' if is_current else ('#ffffff' if is_visited else '#cccccc')
            text_weight = 'bold' if (is_current or is_visited) else 'normal'
            ax.text(x, y, f'{pool_output[i, j]:.2f}', ha='center', va='center',
                   color=text_color, fontsize=10, fontweight=text_weight)
            
            # Draw index number in corner
            idx_color = '#ffffff' if is_current else '#666666'
            ax.text(x - cell_size/2 + 0.15, y + cell_size/2 - 0.15, f'{global_idx}',
                   ha='left', va='top', color=idx_color, fontsize=7, style='italic')
    
    # Draw arrow showing traversal order
    if len(visited_positions) > 1:
        arrow_positions = visited_positions[-5:]  # Show last 5 positions
        for idx in range(len(arrow_positions) - 1):
            i1, j1 = arrow_positions[idx]
            i2, j2 = arrow_positions[idx + 1]
            x1 = matrix_x_offset + j1 * cell_size
            y1 = y_center - i1 * cell_size
            x2 = matrix_x_offset + j2 * cell_size
            y2 = y_center - i2 * cell_size
            
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='#ff6600', alpha=0.8))
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Draw 1D FLATTENED VECTOR (RIGHT SIDE)
    # ═════════════════════════════════════════════════════════════════════════════
    vector_x_offset = matrix_x_offset + w_in * cell_size + spacing
    
    # Draw vertical line (representing the flattened vector)
    for idx in range(total_elements):
        x = vector_x_offset
        y = y_center - idx * cell_size
        
        is_current = (idx == current_idx)
        is_filled = (idx < current_idx)
        
        # Determine colors
        if is_current:
            rect_color = '#ffff00'
            alpha = 0.8
            edge_color = '#ffffff'
            edge_width = 3
        elif is_filled:
            rect_color = '#00ff00'
            alpha = 0.6
            edge_color = '#00ff00'
            edge_width = 2
        else:
            rect_color = '#333333'
            alpha = 0.3
            edge_color = '#666666'
            edge_width = 1
        
        rect = Rectangle((x - cell_size/3 + 0.05, y - cell_size/2 + 0.05),
                       cell_size/1.5 - 0.1, cell_size - 0.1,
                       facecolor=rect_color, alpha=alpha,
                       edgecolor=edge_color, linewidth=edge_width)
        ax.add_patch(rect)
        
        # Draw value in flattened vector if filled
        if idx < current_idx:
            text_color = '#000000' if is_current else '#ffffff'
            text_weight = 'bold' if is_current else 'normal'
            ax.text(x, y, f'{flattened_vector[idx]:.2f}', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight=text_weight)
        elif is_current:
            text_color = '#000000'
            ax.text(x, y, f'{flattened_vector[idx]:.2f}', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')
    
    # Draw connecting arrow from current cell in matrix to vector
    if current_idx < total_elements:
        current_i = current_idx // w_in
        current_j = current_idx % w_in
        x_from = matrix_x_offset + current_j * cell_size
        y_from = y_center - current_i * cell_size
        x_to = vector_x_offset
        y_to = y_center - current_idx * cell_size
        
        # Bezier curve arrow
        ax.annotate('', xy=(x_to - cell_size/3, y_to), xytext=(x_from + cell_size/2, y_from),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#ff6600', alpha=0.9,
                                 connectionstyle="arc3,rad=0.3"))
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Set axis properties
    # ═════════════════════════════════════════════════════════════════════════════
    max_y = max(h_in * cell_size, total_elements * cell_size)
    ax.set_xlim(-1, vector_x_offset + cell_size + 1)
    ax.set_ylim(-max_y - 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    progress = (current_idx / total_elements) * 100 if total_elements > 0 else 0
    title = f'Flatten Layer Animation — Processing Element {current_idx}/{total_elements} ({progress:.1f}% Complete)\n'
    title += f'Reshape: {h_in}×{w_in} → 1D Vector ({total_elements} elements)'
    ax.text(0.5, 1.08, title, transform=ax.transAxes, ha='center', va='bottom',
           fontsize=13, fontweight='bold', color='#ffffff')
    
    # Info box
    info_box = f'Current Value: {flattened_vector[current_idx]:.4f} (from position [{current_idx // w_in}, {current_idx % w_in}])'
    ax.text(0.5, -0.08, info_box, transform=ax.transAxes, ha='center', va='top',
           fontsize=10, color='#ffff00', bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8))
    
    plt.tight_layout()
    return fig

if show_flatten_animation:
    # Flatten the pool output
    flattened_vector = pool_output.flatten()
    total_elements = len(flattened_vector)
    
    anim_placeholder_flatten = st.empty()
    col_info1_flatten, col_info2_flatten = st.columns(2)
    
    with col_info1_flatten:
        info_placeholder_flatten = st.empty()
    
    with col_info2_flatten:
        progress_placeholder_flatten = st.empty()
    
    flatten_anim_delay = anim_delay / 2  # Faster per-element animation
    
    while True:
        for idx in range(total_elements):
            # Draw visualization
            fig = draw_flatten_animation(pool_output, flattened_vector, idx, total_elements, 
                                        idx + 1, anim_speed)
            anim_placeholder_flatten.image(fig_to_pil(fig), width='stretch')
            plt.close(fig)
            
            # Show details
            current_i = idx // pool_output_size
            current_j = idx % pool_output_size
            
            info_text = f"""
### 📍 Flattening Step {idx + 1} / {total_elements}

**Current Position:** [{current_i}, {current_j}]  
**Current Value:** `{flattened_vector[idx]:.4f}`  
**Vector Index:** `{idx}`

---

**Flattened Vector (so far):**
"""
            # Show filled elements
            elements_to_show = min(10, idx + 1)
            for i in range(elements_to_show):
                info_text += f"\n[{i}] = `{flattened_vector[i]:.4f}`"
            
            if total_elements > 10:
                info_text += f"\n... ({total_elements - 10} more elements)"
            
            info_placeholder_flatten.markdown(info_text)
            
            # Progress visualization
            progress_pct = (idx + 1) / total_elements * 100
            progress_text = f"""
### 📈 Transformation Progress

**Shape Transformation:**
- **Input:** {pool_output_size} × {pool_output_size} (2D Matrix)
- **Output:** {total_elements} elements (1D Vector)

**Progress:** {progress_pct:.1f}%

**Completion:** {idx + 1} / {total_elements} elements processed
"""
            progress_placeholder_flatten.markdown(progress_text)
            
            time.sleep(flatten_anim_delay)
        
        info_placeholder_flatten.success(f"✅ **Flatten Complete!**\n\nShape: ({pool_output_size}, {pool_output_size}) → ({total_elements},)\n\nRestarting...")
        time.sleep(2)

else:
    st.info("☑️ Check **Animate Flatten Process** to watch the 2D matrix transform into a 1D vector.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: FULL PIPELINE WITH CUSTOM IMAGE
# ═══════════════════════════════════════════════════════════════════════════════

st.header("🖼️ Full CNN Pipeline: Upload Your Image")

st.markdown("""
Upload any image and watch it transform through the **entire CNN pipeline**:
1. **Convolution** - Extract features with kernel sliding
2. **Pooling** - Reduce spatial dimensions  
3. **Flatten** - Convert to 1D vector for classification

All three stages will be animated **side-by-side** with visual transformations.
""")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "bmp", "gif"], key="pipeline_image")

if uploaded_image is not None:
    # Load and preprocess image
    pil_image = Image.open(uploaded_image)
    
    # Convert to grayscale if needed
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    
    # Resize to fit within processing limits
    max_size = 64
    pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(pil_image, dtype=np.float32) / 255.0
    
    st.subheader("📸 Image Information")
    col_img1, col_img2, col_img3 = st.columns(3)
    
    with col_img1:
        st.metric("Original Size", f"{img_array.shape[0]}×{img_array.shape[1]}")
    
    with col_img2:
        st.metric("Kernel Size", f"{kernel_size}×{kernel_size}")
    
    with col_img3:
        st.metric("Pool Size", f"{pool_size}×{pool_size}")
    
    show_image_pipeline = st.checkbox("Animate Full CNN Pipeline with Pop-up Blocks", value=False, key="pipeline_checkbox")
    
    def apply_convolution_to_image(image, kernel, padding=0, stride=1):
        """Apply convolution operation to image."""
        padded = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
        h, w = padded.shape
        k_h, k_w = kernel.shape
        o_h = ((h - k_h) // stride) + 1
        o_w = ((w - k_w) // stride) + 1
        
        output = np.zeros((o_h, o_w))
        for i in range(0, h - k_h + 1, stride):
            for j in range(0, w - k_w + 1, stride):
                output[i // stride, j // stride] = np.sum(padded[i:i+k_h, j:j+k_w] * kernel)
        
        return output
    
    def apply_pooling_to_image(image, pool_size, pool_stride, pool_type):
        """Apply pooling to image."""
        h, w = image.shape
        o_h = ((h - pool_size) // pool_stride) + 1
        o_w = ((w - pool_size) // pool_stride) + 1
        
        output = np.zeros((o_h, o_w))
        for i in range(0, h - pool_size + 1, pool_stride):
            for j in range(0, w - pool_size + 1, pool_stride):
                window = image[i:i+pool_size, j:j+pool_size]
                if pool_type == "Max Pool":
                    output[i // pool_stride, j // pool_stride] = np.max(window)
                else:
                    output[i // pool_stride, j // pool_stride] = np.mean(window)
        
        return output
    
    def draw_progressive_pipeline(original, conv_output, pool_output, flatten_vec, 
                                  stage, block_idx, total_blocks, conv_shape, pool_shape, kernel_size=3, stride=1, 
                                  kernel_weights=None, pool_size=2, pool_stride=2):
        """Draw progressive blocks with ACTUAL PIXEL VALUES shown during processing."""
        fig = plt.figure(figsize=(28, 10), facecolor='#0a0a0a')
        fig.patch.set_facecolor('#0a0a0a')
        
        # Use GridSpec for better layout with detail panel
        gs = fig.add_gridspec(2, 5, height_ratios=[3, 1], hspace=0.3, wspace=0.25)
        
        # ═════════════════════════════════════════════════════════════════════════════
        # INPUT IMAGE
        # ═════════════════════════════════════════════════════════════════════════════
        ax_input = fig.add_subplot(gs[0, 0])
        ax_input.set_facecolor('#0a0a0a')
        ax_input.imshow(original, cmap='gray', interpolation='nearest')
        ax_input.set_title('INPUT\nOriginal Image', color='#ffffff', fontweight='bold', fontsize=11, pad=10)
        
        # Draw moving kernel box during convolution
        kernel_row, kernel_col = 0, 0
        if stage == 1:
            blocks_per_row = conv_shape[1]
            kernel_row = (block_idx // blocks_per_row) * stride
            kernel_col = (block_idx % blocks_per_row) * stride
            
            rect = Rectangle((kernel_col - 0.5, kernel_row - 0.5), kernel_size, kernel_size,
                           linewidth=3, edgecolor='#ff6600', facecolor='#ff6600', alpha=0.3, linestyle='-')
            ax_input.add_patch(rect)
            
            # Draw grid lines on kernel area
            for i in range(kernel_size + 1):
                ax_input.axhline(y=kernel_row + i - 0.5, xmin=(kernel_col - 0.5)/original.shape[1], 
                               xmax=(kernel_col + kernel_size - 0.5)/original.shape[1], 
                               color='#ff6600', linewidth=0.5, alpha=0.5)
                ax_input.axvline(x=kernel_col + i - 0.5, ymin=1-(kernel_row + kernel_size - 0.5)/original.shape[0], 
                               ymax=1-(kernel_row - 0.5)/original.shape[0], 
                               color='#ff6600', linewidth=0.5, alpha=0.5)
        
        ax_input.axis('off')
        
        # ═════════════════════════════════════════════════════════════════════════════
        # DETAIL PANEL - Show actual computation with pixel values
        # ═════════════════════════════════════════════════════════════════════════════
        ax_detail = fig.add_subplot(gs[1, :3])
        ax_detail.set_facecolor('#1a1a2e')
        ax_detail.set_xlim(0, 10)
        ax_detail.set_ylim(0, 2)
        
        if stage == 1 and kernel_weights is not None:
            # Extract actual pixel values from current kernel position
            h, w = original.shape
            patch = original[kernel_row:min(kernel_row+kernel_size, h), 
                           kernel_col:min(kernel_col+kernel_size, w)]
            
            # Show INPUT PATCH with actual values
            ax_detail.text(0.3, 1.8, 'INPUT PATCH', color='#3498db', fontsize=10, fontweight='bold', va='top')
            
            for i in range(min(kernel_size, patch.shape[0])):
                for j in range(min(kernel_size, patch.shape[1])):
                    val = patch[i, j] if i < patch.shape[0] and j < patch.shape[1] else 0
                    color_intensity = val
                    ax_detail.add_patch(Rectangle((0.1 + j*0.6, 1.4 - i*0.4), 0.55, 0.35, 
                                                 facecolor=plt.cm.gray(color_intensity), 
                                                 edgecolor='#3498db', linewidth=1))
                    ax_detail.text(0.1 + j*0.6 + 0.275, 1.4 - i*0.4 + 0.175, f'{val:.2f}', 
                                 ha='center', va='center', fontsize=7, color='#ffffff' if val < 0.5 else '#000000')
            
            # Show multiplication symbol
            ax_detail.text(2.3, 1.0, '×', color='#ffffff', fontsize=24, fontweight='bold', va='center')
            
            # Show KERNEL with actual weights
            ax_detail.text(2.9, 1.8, 'KERNEL', color='#f39c12', fontsize=10, fontweight='bold', va='top')
            
            for i in range(kernel_size):
                for j in range(kernel_size):
                    val = kernel_weights[i, j]
                    norm_val = (val - kernel_weights.min()) / (kernel_weights.max() - kernel_weights.min() + 1e-10)
                    ax_detail.add_patch(Rectangle((2.7 + j*0.6, 1.4 - i*0.4), 0.55, 0.35, 
                                                 facecolor=plt.cm.RdYlGn(norm_val), 
                                                 edgecolor='#f39c12', linewidth=1))
                    ax_detail.text(2.7 + j*0.6 + 0.275, 1.4 - i*0.4 + 0.175, f'{val:.2f}', 
                                 ha='center', va='center', fontsize=7, color='#000000')
            
            # Show equals and result
            ax_detail.text(4.9, 1.0, '=', color='#ffffff', fontsize=24, fontweight='bold', va='center')
            
            # Calculate actual convolution result
            conv_result = conv_output[block_idx // conv_shape[1], block_idx % conv_shape[1]]
            result_norm = (conv_result - conv_output.min()) / (conv_output.max() - conv_output.min() + 1e-10)
            
            ax_detail.add_patch(Rectangle((5.3, 0.7), 0.8, 0.6, 
                                         facecolor=plt.cm.hot(result_norm), 
                                         edgecolor='#e74c3c', linewidth=3))
            ax_detail.text(5.7, 1.0, f'{conv_result:.3f}', ha='center', va='center', 
                          fontsize=12, fontweight='bold', color='#ffffff')
            
            # Show formula
            ax_detail.text(6.5, 1.0, 'Σ(patch × kernel)', color='#aaaaaa', fontsize=10, va='center', style='italic')
            ax_detail.text(6.5, 0.5, f'Position: ({kernel_row}, {kernel_col}) → Output[{block_idx // conv_shape[1]}, {block_idx % conv_shape[1]}]', 
                          color='#888888', fontsize=9, va='center')
        
        elif stage == 2:
            # POOLING detail
            pool_row = (block_idx // pool_shape[1]) * pool_stride
            pool_col = (block_idx % pool_shape[1]) * pool_stride
            
            ax_detail.text(0.3, 1.8, 'POOLING WINDOW', color='#2ecc71', fontsize=10, fontweight='bold', va='top')
            
            # Extract actual conv values being pooled
            for i in range(pool_size):
                for j in range(pool_size):
                    r, c = pool_row + i, pool_col + j
                    if r < conv_output.shape[0] and c < conv_output.shape[1]:
                        val = conv_output[r, c]
                        val_norm = (val - conv_output.min()) / (conv_output.max() - conv_output.min() + 1e-10)
                        ax_detail.add_patch(Rectangle((0.1 + j*0.8, 1.3 - i*0.5), 0.75, 0.45, 
                                                     facecolor=plt.cm.hot(val_norm), 
                                                     edgecolor='#2ecc71', linewidth=1))
                        ax_detail.text(0.1 + j*0.8 + 0.375, 1.3 - i*0.5 + 0.225, f'{val:.2f}', 
                                     ha='center', va='center', fontsize=9, color='#ffffff')
            
            ax_detail.text(2.1, 1.0, '→ MAX →', color='#2ecc71', fontsize=14, fontweight='bold', va='center')
            
            pool_result = pool_output[block_idx // pool_shape[1], block_idx % pool_shape[1]]
            result_norm = (pool_result - pool_output.min()) / (pool_output.max() - pool_output.min() + 1e-10)
            
            ax_detail.add_patch(Rectangle((3.3, 0.7), 0.8, 0.6, 
                                         facecolor=plt.cm.viridis(result_norm), 
                                         edgecolor='#2ecc71', linewidth=3))
            ax_detail.text(3.7, 1.0, f'{pool_result:.3f}', ha='center', va='center', 
                          fontsize=12, fontweight='bold', color='#ffffff')
            
            ax_detail.text(4.5, 1.0, f'Max pooling at position ({pool_row}, {pool_col})', 
                          color='#888888', fontsize=10, va='center')
        
        elif stage == 3:
            # FLATTEN detail
            ax_detail.text(0.3, 1.8, 'FLATTENING 2D → 1D', color='#9b59b6', fontsize=10, fontweight='bold', va='top')
            
            # Show a few values being converted
            num_show = min(6, len(flatten_vec))
            for i in range(num_show):
                val = flatten_vec[i]
                val_norm = (val - flatten_vec.min()) / (flatten_vec.max() - flatten_vec.min() + 1e-10)
                ax_detail.add_patch(Rectangle((0.1 + i*0.7, 0.8), 0.6, 0.4, 
                                             facecolor=plt.cm.plasma(val_norm), 
                                             edgecolor='#9b59b6', linewidth=1))
                ax_detail.text(0.1 + i*0.7 + 0.3, 1.0, f'{val:.2f}', ha='center', va='center', 
                              fontsize=8, color='#ffffff')
            
            if len(flatten_vec) > num_show:
                ax_detail.text(0.1 + num_show*0.7 + 0.3, 1.0, f'... +{len(flatten_vec)-num_show} more', 
                              color='#888888', fontsize=9, va='center')
            
            ax_detail.text(6, 1.0, f'Total: {len(flatten_vec)} neurons', color='#9b59b6', fontsize=12, 
                          fontweight='bold', va='center')
        
        else:
            ax_detail.text(5, 1, 'Processing Complete', color='#2ecc71', fontsize=14, 
                          fontweight='bold', ha='center', va='center')
        
        ax_detail.axis('off')
        ax_detail.set_title('🔍 DETAIL VIEW - Actual Values', color='#ffffff', fontsize=10, pad=5)
        
        # ═════════════════════════════════════════════════════════════════════════════
        # CONVOLUTION LAYER - BLOCKS POP UP
        # ═════════════════════════════════════════════════════════════════════════════
        ax_conv = fig.add_subplot(gs[0, 1])
        ax_conv.set_facecolor('#0a0a0a')
        
        # Draw populated convolution blocks
        if stage >= 1:
            conv_display = (conv_output - np.min(conv_output)) / (np.max(conv_output) - np.min(conv_output) + 1e-10)
            
            if stage == 1:
                # Show only blocks that have been created so far
                partial_conv = np.zeros_like(conv_output)
                blocks_created = min(block_idx + 1, conv_output.size)
                for i in range(blocks_created):
                    row = i // conv_shape[1]
                    col = i % conv_shape[1]
                    if row < conv_shape[0] and col < conv_shape[1]:
                        partial_conv[row, col] = conv_display[row, col]
                ax_conv.imshow(partial_conv, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
            else:
                ax_conv.imshow(conv_display, cmap='hot', interpolation='nearest')
        
        ax_conv.set_title(f'CONVOLUTION\n{conv_shape[0]}×{conv_shape[1]} blocks', 
                         color='#ffff00' if stage == 1 else '#cccccc', fontweight='bold', fontsize=11, pad=10)
        ax_conv.axis('off')
        
        if stage == 1:
            progress = f"{block_idx + 1}/{conv_shape[0]*conv_shape[1]} blocks"
            ax_conv.text(0.5, -0.15, progress, transform=ax_conv.transAxes, ha='center', 
                        color='#ffff00', fontweight='bold', fontsize=10)
        
        # ═════════════════════════════════════════════════════════════════════════════
        # POOLING LAYER - BLOCKS POP UP
        # ═════════════════════════════════════════════════════════════════════════════
        ax_pool = fig.add_subplot(gs[0, 2])
        ax_pool.set_facecolor('#0a0a0a')
        
        if stage >= 2:
            pool_display = (pool_output - np.min(pool_output)) / (np.max(pool_output) - np.min(pool_output) + 1e-10)
            
            if stage == 2:
                # Show only blocks that have been created so far
                partial_pool = np.zeros_like(pool_output)
                blocks_created = min(block_idx + 1, pool_output.size)
                for i in range(blocks_created):
                    row = i // pool_shape[1]
                    col = i % pool_shape[1]
                    if row < pool_shape[0] and col < pool_shape[1]:
                        partial_pool[row, col] = pool_display[row, col]
                ax_pool.imshow(partial_pool, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
            else:
                ax_pool.imshow(pool_display, cmap='viridis', interpolation='nearest')
        
        ax_pool.set_title(f'POOLING\n{pool_shape[0]}×{pool_shape[1]} blocks', 
                         color='#00ff00' if stage == 2 else '#cccccc', fontweight='bold', fontsize=11, pad=10)
        ax_pool.axis('off')
        
        if stage == 2:
            progress = f"{block_idx + 1}/{pool_shape[0]*pool_shape[1]} blocks"
            ax_pool.text(0.5, -0.15, progress, transform=ax_pool.transAxes, ha='center', 
                        color='#00ff00', fontweight='bold', fontsize=10)
        
        # ═════════════════════════════════════════════════════════════════════════════
        # FLATTEN + DENSE LAYER
        # ═════════════════════════════════════════════════════════════════════════════
        ax_flatten = fig.add_subplot(gs[0, 3])
        ax_flatten.set_facecolor('#0a0a0a')
        
        # Show flattening progress - vertical bar visualization
        if stage >= 3:
            ax_flatten.set_xlim(-0.5, 2)
            ax_flatten.set_ylim(-0.5, len(flatten_vec) + 0.5)
            
            if stage == 3:
                # Show neurons popping up during flatten
                neurons_created = min(block_idx + 1, len(flatten_vec))
                for i in range(neurons_created):
                    y_pos = len(flatten_vec) - 1 - i
                    x_pos = 1
                    neuron_color = plt.cm.plasma(flatten_vec[i] if len(flatten_vec) > 0 else 0)
                    
                    ax_flatten.scatter(x_pos, y_pos, s=100, c=[neuron_color], 
                                     edgecolors='#00ffff', linewidth=1.5, alpha=0.8, zorder=10)
            else:
                # Stage 4+: Show all flattened neurons
                for i in range(len(flatten_vec)):
                    y_pos = len(flatten_vec) - 1 - i
                    x_pos = 1
                    neuron_color = plt.cm.plasma(flatten_vec[i] if len(flatten_vec) > 0 else 0)
                    
                    ax_flatten.scatter(x_pos, y_pos, s=100, c=[neuron_color], 
                                     edgecolors='#00ffff', linewidth=1, alpha=0.6, zorder=10)
        
        ax_flatten.set_title('FLATTEN\n1D Vector', 
                           color='#ffff00' if stage == 3 else '#cccccc', fontweight='bold', fontsize=11, pad=10)
        ax_flatten.set_xticks([])
        ax_flatten.set_yticks([])
        ax_flatten.axis('off')
        
        if stage == 3:
            progress = f"Neurons: {block_idx + 1}/{len(flatten_vec)}"
            ax_flatten.text(0.5, -0.1, progress, transform=ax_flatten.transAxes, ha='center', 
                          color='#ffff00', fontweight='bold', fontsize=9)
        
        # ═════════════════════════════════════════════════════════════════════════════
        # DENSE LAYER - FULLY CONNECTED NEURONS
        # ═════════════════════════════════════════════════════════════════════════════
        ax_dense = fig.add_subplot(gs[0, 4])
        ax_dense.set_facecolor('#0a0a0a')
        
        if stage >= 4:
            # Create 8 dense layer neurons connected to flattened vector
            dense_neurons = 8
            dense_values = np.random.RandomState(42).rand(dense_neurons) * 0.8  # Consistent values
            
            ax_dense.set_xlim(-0.5, 2.5)
            ax_dense.set_ylim(-0.5, dense_neurons + 0.5)
            
            if stage == 4:
                # Show neurons popping up during dense layer
                neurons_created = min(block_idx, dense_neurons)
                
                for i in range(neurons_created):
                    y_pos = dense_neurons - 1 - i
                    x_pos = 1
                    neuron_color = plt.cm.Spectral(i / dense_neurons)
                    neuron_size = 400
                    
                    ax_dense.scatter(x_pos, y_pos, s=neuron_size, c=[neuron_color], 
                                   edgecolors='#00ff00', linewidth=2, alpha=0.8, zorder=10)
                    ax_dense.text(x_pos + 0.5, y_pos, f'H{i+1}', 
                                 ha='left', va='center', color='#ffffff', fontsize=9, fontweight='bold')
            elif stage == 4:
                # Stage 4: Show all dense neurons
                for i in range(dense_neurons):
                    y_pos = dense_neurons - 1 - i
                    x_pos = 1
                    neuron_color = plt.cm.Spectral(i / dense_neurons)
                    
                    ax_dense.scatter(x_pos, y_pos, s=500, c=[neuron_color], 
                                   edgecolors='#00ff00', linewidth=2, alpha=0.8, zorder=10)
                    ax_dense.text(x_pos + 0.5, y_pos, f'H{i+1}: {dense_values[i]:.2f}', 
                                 ha='left', va='center', color='#ffffff', fontsize=9, fontweight='bold')
            else:
                # Stage 5: Show OUTPUT LAYER with winner only - 8 classes
                num_classes = 8
                chunk_size = len(flatten_vec) // num_classes
                neuron_values = [np.mean(flatten_vec[i*chunk_size:(i+1)*chunk_size]) for i in range(num_classes)]
                neuron_labels = [f'Class {i+1}' for i in range(num_classes)]
                selected_neuron = np.argmax(neuron_values)
                
                ax_dense.set_xlim(-0.5, 2.5)
                ax_dense.set_ylim(-0.5, num_classes + 0.5)
                
                # Show all 8 output classes
                for i in range(num_classes):
                    x_pos = 1
                    y_pos = num_classes - 0.5 - i * 1.0
                    neuron_color = plt.cm.Set3(i)
                    is_winner = (i == selected_neuron)
                    neuron_size = 700 if is_winner else 500
                    
                    ax_dense.scatter(x_pos, y_pos, s=neuron_size, c=[neuron_color], 
                                   edgecolors='#ffff00' if is_winner else '#666666', 
                                   linewidth=4 if is_winner else 1.5, 
                                   alpha=1.0 if is_winner else 0.4, zorder=10)
                    
                    # Add labels and values
                    value_text = f"{neuron_labels[i]}\n{neuron_values[i]:.3f}"
                    ax_dense.text(x_pos + 0.5, y_pos, value_text, 
                                ha='left', va='center', 
                                color='#ffff00' if is_winner else '#888888', 
                                fontsize=10 if is_winner else 8, 
                                fontweight='bold' if is_winner else 'normal')
                    
                    # Add winner crown
                    if is_winner:
                        ax_dense.text(x_pos - 0.8, y_pos + 0.5, '👑', fontsize=20, ha='center')
                        circle = plt.Circle((x_pos, y_pos), 0.4, color='#ffff00', fill=False, 
                                          linewidth=2.5, linestyle='--', alpha=0.7)
                        ax_dense.add_patch(circle)
        
        # Update title based on stage
        if stage <= 4:
            title = 'DENSE LAYER\n8 Hidden Neurons'
            title_color = '#00ff00' if stage == 4 else '#cccccc'
        else:
            title = '🏆 OUTPUT LAYER\n8 Classes'
            title_color = '#ffff00'
        
        ax_dense.set_title(title, color=title_color, fontweight='bold', fontsize=11, pad=10)
        ax_dense.set_xticks([])
        ax_dense.set_yticks([])
        ax_dense.axis('off')
        
        if stage == 4:
            progress = f"Building dense layer..."
            ax_dense.text(0.5, -0.1, progress, transform=ax_dense.transAxes, ha='center', 
                         color='#00ff00', fontweight='bold', fontsize=10)
        
        # ═════════════════════════════════════════════════════════════════════════════
        # OUTPUT LAYER - 3 CLASSIFICATION CLASSES (shown in DENSE panel at stage 5)
        # ═════════════════════════════════════════════════════════════════════════════
        
        plt.tight_layout()
        return fig
    
    if show_image_pipeline:
        # Apply all transformations
        conv_output_img = apply_convolution_to_image(img_array, kernel, padding, stride)
        pool_output_img = apply_pooling_to_image(conv_output_img, pool_size, pool_stride, pool_type)
        flatten_vec_img = pool_output_img.flatten()
        
        conv_shape = conv_output_img.shape
        pool_shape = pool_output_img.shape
        
        anim_placeholder_pipeline = st.empty()
        info_placeholder_pipe = st.empty()
        
        pipeline_delay = 0.08  # Smooth and slower animation
        
        while True:
            # Stage 1: Convolution blocks pop up
            conv_total = conv_shape[0] * conv_shape[1]
            skip_frames = max(1, conv_total // 25)  # Render more frames for smooth animation
            for block_idx in range(conv_total):
                if block_idx % skip_frames == 0 or block_idx == conv_total - 1:  # Render key frames
                    fig = draw_progressive_pipeline(img_array, conv_output_img, pool_output_img, flatten_vec_img, 
                                                   1, block_idx, conv_total, conv_shape, pool_shape,
                                                   kernel_size=kernel_size, stride=stride, kernel_weights=kernel,
                                                   pool_size=pool_size, pool_stride=pool_stride)
                    anim_placeholder_pipeline.image(fig_to_pil(fig), width='stretch')
                    plt.close(fig)
                    
                    info_placeholder_pipe.markdown(f"""
### 🔵 STAGE 1: CONVOLUTION
**Blocks Popping Up:** {block_idx + 1}/{conv_total}

The convolutional kernel slides across the image and creates feature maps.
Each new block represents extracted features from different image regions.
                    """)
                    time.sleep(pipeline_delay)
            
            # Stage 2: Pooling blocks pop up
            pool_total = pool_shape[0] * pool_shape[1]
            skip_frames = max(1, pool_total // 25)
            for block_idx in range(pool_total):
                if block_idx % skip_frames == 0 or block_idx == pool_total - 1:
                    fig = draw_progressive_pipeline(img_array, conv_output_img, pool_output_img, flatten_vec_img, 
                                                   2, block_idx, pool_total, conv_shape, pool_shape,
                                                   kernel_size=kernel_size, stride=stride, kernel_weights=kernel,
                                                   pool_size=pool_size, pool_stride=pool_stride)
                    anim_placeholder_pipeline.image(fig_to_pil(fig), width='stretch')
                    plt.close(fig)
                    
                    info_placeholder_pipe.markdown(f"""
### 🟢 STAGE 2: POOLING
**Blocks Popping Up:** {block_idx + 1}/{pool_total}

Pooling reduces spatial dimensions by selecting max/average values.
Fewer but more important features are retained from the convolution output.
                    """)
                    time.sleep(pipeline_delay)
            
            # Stage 3: Flatten neurons pop up  
            flatten_total = len(flatten_vec_img)
            skip_frames = max(1, flatten_total // 3)  # Only ~3 frames total for speed
            progress_increment = skip_frames  # Match progress to actual frame jumps
            for block_idx in range(flatten_total):
                if block_idx % skip_frames == 0 or block_idx == flatten_total - 1:
                    fig = draw_progressive_pipeline(img_array, conv_output_img, pool_output_img, flatten_vec_img, 
                                                   3, block_idx, flatten_total, conv_shape, pool_shape,
                                                   kernel_size=kernel_size, stride=stride, kernel_weights=kernel,
                                                   pool_size=pool_size, pool_stride=pool_stride)
                    anim_placeholder_pipeline.image(fig_to_pil(fig), width='stretch')
                    plt.close(fig)
                    
                    # Show progress in 300-neuron increments
                    display_progress = ((block_idx + progress_increment) // progress_increment) * progress_increment
                    display_progress = min(display_progress, flatten_total)  # Cap at total
                    
                    info_placeholder_pipe.markdown(f"""
### 🟡 STAGE 3: FLATTEN → OUTPUT NEURONS
**Neurons Added:** {display_progress}/{flatten_total}

The 2D pooling output is flattened into a 1D vector.
All neurons connect to classification layers for final prediction.
                    """)
                    time.sleep(pipeline_delay)
            
            # Stage 4: Dense layer neurons pop up
            dense_total = 8
            skip_frames_dense = max(1, dense_total // 5)
            for block_idx in range(dense_total):
                if block_idx % skip_frames_dense == 0 or block_idx == dense_total - 1:
                    fig = draw_progressive_pipeline(img_array, conv_output_img, pool_output_img, flatten_vec_img, 
                                                   4, block_idx, dense_total, conv_shape, pool_shape,
                                                   kernel_size=kernel_size, stride=stride, kernel_weights=kernel,
                                                   pool_size=pool_size, pool_stride=pool_stride)
                    anim_placeholder_pipeline.image(fig_to_pil(fig), width='stretch')
                    plt.close(fig)
                    
                    info_placeholder_pipe.markdown(f"""
### 🟢 STAGE 4: DENSE LAYER
**Hidden Neurons Created:** {block_idx + 1}/{dense_total}

Dense layer applies learned weights and biases to extracted features.
These 8 hidden neurons process and compress the information.
                    """)
                    time.sleep(pipeline_delay)
            
            # Stage 5: Output layer - show final winner directly (no animation)
            fig = draw_progressive_pipeline(img_array, conv_output_img, pool_output_img, flatten_vec_img, 
                                           5, 0, 1, conv_shape, pool_shape,
                                           kernel_size=kernel_size, stride=stride, kernel_weights=kernel,
                                           pool_size=pool_size, pool_stride=pool_stride)
            anim_placeholder_pipeline.image(fig_to_pil(fig), width='stretch')
            plt.close(fig)
            
            info_placeholder_pipe.markdown(f"""
### 🏆 STAGE 5: OUTPUT LAYER - CLASSIFICATION WINNER
**Final Prediction Selected!**

The dense layer connects to 8 output neurons (one per class).
The neuron with highest activation is the winner (marked with 👑)!
            """)
            time.sleep(pipeline_delay * 2)
            
            info_placeholder_pipe.success(f"""
✅ **COMPLETE CNN PIPELINE VISUALIZATION!**

**Full Network Architecture:**
- **Input:** {img_array.shape[0]}×{img_array.shape[1]} image
- **Convolution:** {conv_shape[0]}×{conv_shape[1]} feature maps
- **Pooling:** {pool_shape[0]}×{pool_shape[1]} reduced maps  
- **Flatten:** {flatten_total} vector elements
- **Dense Layer:** 8 hidden neurons with learnable weights
- **Output Layer:** 8 classification neurons (Classes 1-8)
- **Prediction:** Highest activation = winner class! 🏆

Restarting animation...
            """)
            time.sleep(0.1)
    
    else:
        st.info("☑️ Check **Animate Full CNN Pipeline with Pop-up Blocks** to watch blocks progressively pop up through each layer!")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: MANIM VIDEO GENERATION - Professional Animation
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("🎬 Professional Manim Animation")

st.markdown("""
Generate a **professional video animation** showing how your image flows through the CNN using **Manim** 
(the same animation library used by 3Blue1Brown).
""")

# Artistic Animation Info
st.markdown("""
**Artistic Neural Network Animation** - Inspired by Kim Seonghyun's Design Korea 2024 artwork:
- 🌟 Flowing, organic neural network structure
- ✨ Glowing nodes with pulsing data flow
- 🎆 Beautiful color transitions and effects
- 🏆 Dramatic classification reveal
""")

manim_image = st.file_uploader("Upload Image for Manim Animation", type=["jpg", "jpeg", "png", "bmp", "gif"], key="manim_image")

if manim_image is not None:
    import subprocess
    import os
    
    # Save the uploaded image
    temp_image_path = os.path.join(os.path.dirname(__file__), "temp_input.png")
    
    pil_img = Image.open(manim_image)
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
    pil_img.thumbnail((32, 32), Image.Resampling.LANCZOS)
    pil_img.save(temp_image_path)
    
    col_preview, col_settings = st.columns([1, 2])
    
    with col_preview:
        st.image(pil_img, caption="Preview (32×32)", width=150)
    
    with col_settings:
        # Prediction class - what the CNN should "predict"
        winner_class = st.text_input("🏆 Prediction Result", value="Sonic", 
                                      help="What class should the CNN predict? (e.g., Sonic, Mario, Cat)")
        
        # Other classes for comparison
        other_classes = st.text_input("Other Classes (comma-separated)", value="Mario,Pikachu,Kirby",
                                       help="Other classes to show in the output layer")
        
        quality = st.selectbox("Video Quality", ["Low (Fast)", "Medium", "High (Slow)"], index=0)
        quality_flag = "-ql" if "Low" in quality else ("-qm" if "Medium" in quality else "-qh")
        
        generate_video = st.button("🎬 Generate Manim Animation", type="primary")
    
    if generate_video:
        video_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        progress_placeholder.info("🎬 Generating animation... This takes about 15-30 seconds...")
        
        try:
            # Set environment variables for image path and class names
            env = os.environ.copy()
            env['CNN_IMAGE_PATH'] = temp_image_path
            
            # Build class names list - winner first, then others
            all_classes = [winner_class.strip()] + [c.strip() for c in other_classes.split(',') if c.strip()]
            env['CNN_CLASS_NAMES'] = ','.join(all_classes[:4])  # Max 4 classes
            env['CNN_WINNER_CLASS'] = winner_class.strip()
            
            # Run manim (no -p flag to avoid opening preview window)
            script_dir = os.path.dirname(__file__)
            
            # Use artistic animation
            manim_script = os.path.join(script_dir, "cnn_artistic.py")
            scene_name = "ArtisticNeuralNetwork"
            video_subfolder = "cnn_artistic"
            
            # Find manim executable in same location as python
            import sys
            python_dir = os.path.dirname(sys.executable)
            manim_exe = os.path.join(python_dir, "manim.exe") if os.name == 'nt' else os.path.join(python_dir, "manim")
            if not os.path.exists(manim_exe):
                manim_exe = "manim"  # Fall back to PATH
            
            result = subprocess.run(
                [manim_exe, quality_flag, "--disable_caching", manim_script, scene_name],
                capture_output=True,
                text=True,
                cwd=script_dir,
                env=env,
                timeout=180  # 3 minute timeout for uncached render
            )
            
            if result.returncode == 0:
                # Find the generated video
                media_dir = os.path.join(script_dir, "media", "videos", video_subfolder)
                
                # Find the quality folder
                quality_folders = {
                    "-ql": "480p15",
                    "-qm": "720p30", 
                    "-qh": "1080p60"
                }
                video_folder = os.path.join(media_dir, quality_folders.get(quality_flag, "480p15"))
                video_path = os.path.join(video_folder, f"{scene_name}.mp4")
                
                if os.path.exists(video_path):
                    progress_placeholder.success("✅ Animation generated successfully!")
                    
                    # Display video
                    with open(video_path, "rb") as video_file:
                        video_bytes = video_file.read()
                        video_placeholder.video(video_bytes)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Video",
                        data=video_bytes,
                        file_name="cnn_animation.mp4",
                        mime="video/mp4"
                    )
                else:
                    progress_placeholder.warning(f"Video generated but file not found at expected path. Check: {video_folder}")
                    st.code(result.stdout)
            else:
                progress_placeholder.error("❌ Error generating animation")
                st.code(f"STDERR: {result.stderr}\n\nSTDOUT: {result.stdout}")
                
        except subprocess.TimeoutExpired:
            progress_placeholder.error("❌ Animation generation timed out (>3 minutes)")
        except FileNotFoundError:
            progress_placeholder.error("❌ Manim not found. Install with: pip install manim")
        except Exception as e:
            progress_placeholder.error(f"❌ Error: {str(e)}")

else:
    st.info("📤 Upload an image above to generate a professional Manim animation showing CNN processing.")
