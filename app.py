import streamlit as st
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
from skimage.feature import hog, local_binary_pattern as _lbp_fn
from skimage import exposure
from PIL import Image
from matplotlib.animation import FuncAnimation, PillowWriter
import io
import tempfile
import os as _os

st.set_page_config(page_title="Feature Descriptor Visualizer", layout="wide")
st.title("🔬 Feature Descriptor – Step-by-Step Visualizer")
_mode_choice = st.radio(
    "**Choose a descriptor to explore:**",
    ["HOG – Histogram of Oriented Gradients", "LBP – Local Binary Pattern"],
    horizontal=True,
)
mode = "HOG" if _mode_choice.startswith("HOG") else "LBP"
st.markdown("---")

# ── Sidebar controls ──────────────────────────────────────────────────────────
resize_to = st.sidebar.selectbox("Resize image to", [64, 128, 256, 512], index=1)
st.sidebar.markdown("---")
if mode == "HOG":
    st.sidebar.header("⚙️ HOG Parameters")
    pixels_per_cell = st.sidebar.slider("Pixels per Cell", min_value=4, max_value=32, value=8, step=4)
    cells_per_block  = st.sidebar.slider("Cells per Block",  min_value=1, max_value=4,  value=2, step=1)
    orientations     = st.sidebar.slider("Orientations (bins)", min_value=4, max_value=18, value=9, step=1)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 What is HOG?")
    st.sidebar.markdown(
        "HOG captures **edge directions** by computing gradient orientations in local **cells**, "
        "then normalises them over **blocks** to produce a robust descriptor used widely in "
        "object detection (e.g. pedestrian detection)."
    )
else:  # LBP
    st.sidebar.header("⚙️ LBP Parameters")
    lbp_P      = st.sidebar.slider("P – Neighbours on circle", min_value=4, max_value=24, value=8, step=4)
    lbp_R      = st.sidebar.slider("R – Radius (pixels)",      min_value=1, max_value=5,  value=1, step=1)
    lbp_method = st.sidebar.selectbox(
        "Method",
        ["uniform", "default", "ror", "nri_uniform", "var"],
        index=0,
        help=(
            "uniform: ≤2 bit transitions → P+2 bins  |  "
            "default: raw binary → 2ᴾ codes  |  "
            "ror: rotation-invariant  |  "
            "nri_uniform: non-rotation-invariant uniform  |  "
            "var: neighbourhood variance (continuous)"
        ),
    )
    grid_y = 4
    grid_x = 4
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 What is LBP?")
    st.sidebar.markdown(
        "LBP encodes **local texture** by comparing each pixel to its **P** circularly-placed "
        "neighbours at radius **R**. The binary code → decimal index. "
        "Histograms of these codes form a powerful texture descriptor used in "
        "face detection, texture recognition and more."
    )

# ── Image upload ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("⬆️ Upload an image to start the HOG walkthrough.")
    st.stop()

# Load and preprocess
pil_img  = Image.open(uploaded).convert("RGB")
img_rgb  = np.array(pil_img)
img_resized = cv2.resize(img_rgb, (resize_to, resize_to))
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: render a matplotlib figure to a PIL image
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return Image.open(buf)


# =============================================================================
# LBP PIPELINE  (runs when mode == "LBP", calls st.stop() at end)
# =============================================================================
if mode == "LBP":
    st.markdown(
        "Explore each stage of the **Local Binary Pattern (LBP)** pipeline. "
        "LBP is a highly efficient texture descriptor widely used in face detection and texture analysis."
    )

    # ── bilinear interpolation helper ────────────────────────────────────────
    def _bilinear(img, y, x):
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        y1 = min(y0 + 1, img.shape[0] - 1)
        x1 = min(x0 + 1, img.shape[1] - 1)
        y0, x0 = max(y0, 0), max(x0, 0)
        dy, dx = y - y0, x - x0
        return (img[y0, x0] * (1 - dy) * (1 - dx) +
                img[y0, x1] * (1 - dy) * dx +
                img[y1, x0] * dy * (1 - dx) +
                img[y1, x1] * dy * dx)

    # =========================================================================
    # LBP STEP 1 – Original & Grayscale
    # =========================================================================
    st.markdown("---")
    st.header("Step 1 · Original Image & Grayscale Conversion")
    st.markdown(
        "LBP operates on **grayscale** intensity values. "
        "Each pixel will be compared with its P circular neighbours at radius R."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original (RGB)")
        st.image(img_resized, caption=f"Resized to {resize_to}×{resize_to}", width='stretch')
    with c2:
        st.subheader("Grayscale")
        st.image(img_gray.astype(np.uint8), caption="Grayscale", clamp=True, width='stretch')

    # =========================================================================
    # LBP STEP 2 – Circular neighbourhood & binary encoding (demo pixel)
    # =========================================================================
    st.markdown("---")
    st.header("Step 2 · Circular Neighbourhood & Binary Encoding")
    st.markdown(f"""
**How a single pixel gets its LBP code (P={lbp_P}, R={lbp_R}):**

1. Place **P = {lbp_P}** sample points on a circle of **radius R = {lbp_R}** around the centre pixel.
2. Use **bilinear interpolation** for non-integer neighbour positions.
3. Threshold each neighbour $g_p$ against the centre value $g_c$:
$$s(g_p - g_c) = \\begin{{cases}} 1 & g_p \\geq g_c \\\\ 0 & g_p < g_c \\end{{cases}}$$
4. Concatenate P bits counter-clockwise → convert to decimal → **LBP code** of that pixel.
""")

    cy_d = img_gray.shape[0] // 2
    cx_d = img_gray.shape[1] // 2
    c_val_d = float(img_gray[cy_d, cx_d])
    ang_d = [2 * np.pi * p / lbp_P for p in range(lbp_P)]
    nx_d  = [cx_d + lbp_R * np.cos(a) for a in ang_d]
    ny_d  = [cy_d - lbp_R * np.sin(a) for a in ang_d]
    nv_d  = [_bilinear(img_gray, ny_d[p], nx_d[p]) for p in range(lbp_P)]
    bits_d = [1 if v >= c_val_d else 0 for v in nv_d]
    code_d = sum(b * (2 ** i) for i, b in enumerate(bits_d))

    zoom_r = max(lbp_R * 4, 14)
    y0_sub = max(0, cy_d - zoom_r)
    x0_sub = max(0, cx_d - zoom_r)
    sub_g   = img_gray[y0_sub: cy_d + zoom_r + 1, x0_sub: cx_d + zoom_r + 1]
    off_y   = cy_d - y0_sub
    off_x   = cx_d - x0_sub

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    # left: zoomed neighbourhood
    axes2[0].imshow(sub_g, cmap="gray", vmin=0, vmax=255,
                    extent=[-off_x - 0.5, sub_g.shape[1] - off_x - 0.5,
                             sub_g.shape[0] - off_y - 0.5, -off_y - 0.5])
    circ2 = plt.Circle((0, 0), lbp_R, color="yellow", fill=False, lw=2)
    axes2[0].add_patch(circ2)
    axes2[0].plot(0, 0, "r*", markersize=14)
    axes2[0].annotate(f"c={int(c_val_d)}", (0, 0), xytext=(0.4, 0.4), color="red", fontsize=9)
    for p in range(lbp_P):
        rx, ry = nx_d[p] - cx_d, ny_d[p] - cy_d
        col = "lime" if bits_d[p] else "tomato"
        axes2[0].plot(rx, ry, "o", color=col, markersize=9)
        axes2[0].annotate(
            f"{int(nv_d[p])}\nb{p}={bits_d[p]}",
            (rx, ry), textcoords="offset points", xytext=(4, 4),
            fontsize=6, color=col
        )
    axes2[0].set_xlim(-off_x - 0.5, sub_g.shape[1] - off_x - 0.5)
    axes2[0].set_ylim(sub_g.shape[0] - off_y - 0.5, -off_y - 0.5)
    axes2[0].set_title(
        f"Centre pixel ({cy_d},{cx_d}), value = {int(c_val_d)}\n"
        f"Bits (b0…b{lbp_P-1}): {''.join(str(b) for b in bits_d)}  →  LBP = {code_d}",
        fontsize=9
    )
    axes2[0].axis("off")
    # right: bit bar chart
    axes2[1].bar(range(lbp_P), bits_d,
                 color=["lime" if b else "tomato" for b in bits_d], width=0.85, edgecolor="white")
    axes2[1].set_xticks(range(lbp_P))
    axes2[1].set_xticklabels([f"b{i}" for i in range(lbp_P)], fontsize=8)
    axes2[1].set_yticks([0, 1]); axes2[1].set_yticklabels(["0 (g_p < c)", "1 (g_p ≥ c)"])
    axes2[1].set_title(f"Binary bits for demo pixel  →  decimal = {code_d}")
    axes2[1].set_xlabel("Bit index p (counter-clockwise around circle)")
    plt.tight_layout()
    st.image(fig_to_pil(fig2), width='stretch')
    plt.close(fig2)

    # =========================================================================
    # LBP STEP 3 – Animated encoding walkthrough
    # =========================================================================
    st.markdown("---")
    st.header("Step 3 · Animated Binary Pattern Encoding")
    st.markdown(
        "Watch LBP codes being computed pixel-by-pixel across a central patch. "
        "**Green** neighbour ≥ centre → bit = 1 · **Red** → bit = 0."
    )
    PATCH_SZ = 32
    py0_lbp  = img_gray.shape[0] // 2 - PATCH_SZ // 2
    px0_lbp  = img_gray.shape[1] // 2 - PATCH_SZ // 2
    patch_g  = img_gray[py0_lbp: py0_lbp + PATCH_SZ, px0_lbp: px0_lbp + PATCH_SZ]
    stride3  = max(1, (PATCH_SZ - 2 * lbp_R) // 6)
    pix3     = [(r, c)
                for r in range(lbp_R, PATCH_SZ - lbp_R, stride3)
                for c in range(lbp_R, PATCH_SZ - lbp_R, stride3)][:42]

    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4.5))
    plt.tight_layout(pad=2)
    axes3[0].imshow(patch_g, cmap="gray", vmin=0, vmax=255)
    rect3   = patches.Rectangle((0, 0), 1, 1, lw=2, edgecolor="yellow", facecolor="none")
    axes3[0].add_patch(rect3)
    axes3[0].axis("off")
    title3  = axes3[0].set_title("Patch")
    ZOOM3   = max(lbp_R * 8 + 4, 20)
    zoom_im3 = axes3[1].imshow(np.zeros((ZOOM3, ZOOM3)), cmap="gray", vmin=0, vmax=255)
    axes3[1].axis("off")
    axes3[1].set_title("Neighbourhood (zoomed)")
    bars3   = axes3[2].bar(range(lbp_P), np.zeros(lbp_P), color="steelblue",
                           width=0.85, edgecolor="white")
    axes3[2].set_ylim(-0.1, 1.6)
    axes3[2].set_xticks(range(lbp_P))
    axes3[2].set_xticklabels([str(i) for i in range(lbp_P)], fontsize=6)
    axes3[2].set_xlabel("Bit index p")
    title_b3 = axes3[2].set_title("Bits")

    def _upd_lbp3(fi):
        r, c = pix3[fi % len(pix3)]
        rect3.set_xy((c - 0.5, r - 0.5))
        title3.set_text(f"px ({r},{c}) val={int(patch_g[r, c])}")
        # zoomed neighbourhood
        y0z = max(0, r - lbp_R - 1);  x0z = max(0, c - lbp_R - 1)
        sub  = patch_g[y0z: r + lbp_R + 2, x0z: c + lbp_R + 2]
        scale = max(1, ZOOM3 // max(sub.shape[0], sub.shape[1], 1))
        zoomed = np.kron(sub, np.ones((scale, scale)))
        pad_h  = max(0, ZOOM3 - zoomed.shape[0])
        pad_w  = max(0, ZOOM3 - zoomed.shape[1])
        zoomed = np.pad(zoomed, ((0, pad_h), (0, pad_w)), mode="edge")
        zoom_im3.set_data(zoomed[:ZOOM3, :ZOOM3])
        # compute bits
        cv  = float(patch_g[r, c])
        bv  = []
        for p in range(lbp_P):
            a  = 2 * np.pi * p / lbp_P
            yn = r - lbp_R * np.sin(a)
            xn = c + lbp_R * np.cos(a)
            bv.append(1 if _bilinear(patch_g, yn, xn) >= cv else 0)
        for bar, b in zip(bars3, bv):
            bar.set_height(b)
            bar.set_color("lime" if b else "tomato")
        lbp_v = sum(b * (2 ** i) for i, b in enumerate(bv))
        title_b3.set_text(f"Bits → LBP = {lbp_v}")
        return [zoom_im3, rect3, title3, title_b3, *bars3]

    anim_lbp3 = FuncAnimation(fig3, _upd_lbp3, frames=len(pix3), interval=400, blit=False)
    tmp_lbp3  = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp_lbp3.close()
    anim_lbp3.save(tmp_lbp3.name, writer=PillowWriter(fps=3))
    plt.close(fig3)
    with open(tmp_lbp3.name, "rb") as f:
        st.image(f.read(), width='stretch')
    _os.unlink(tmp_lbp3.name)

    # =========================================================================
    # LBP STEP 4 – Full LBP image
    # =========================================================================
    st.markdown("---")
    st.header("Step 4 · Full LBP-Encoded Image")
    lbp_img = _lbp_fn(img_gray, lbp_P, lbp_R, method=lbp_method)
    lbp_disp = ((lbp_img - lbp_img.min()) /
                (lbp_img.max() - lbp_img.min() + 1e-8) * 255).astype(np.uint8)
    _method_notes = {
        "default":     f"Raw decimal encoding — {2**lbp_P} possible codes (2ᴾ)",
        "ror":         "Rotation-invariant — each code mapped to its minimum bit rotation",
        "uniform":     f"Only patterns with ≤ 2 bit transitions kept → {lbp_P + 2} bins (P+2)",
        "nri_uniform": f"Non-rotation-invariant uniform → {lbp_P*(lbp_P-1)+3} bins",
        "var":         "Variance of the neighbourhood — continuous values, captures contrast",
    }
    st.info(f"**Method `{lbp_method}`**: {_method_notes.get(lbp_method, '')}")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Grayscale Input")
        st.image(img_gray.astype(np.uint8), clamp=True, width='stretch')
    with c2:
        st.subheader("LBP Image (normalised for display)")
        st.image(lbp_disp, clamp=True, width='stretch')
    fig4b, ax4b = plt.subplots(figsize=(10, 3))
    im4b = ax4b.imshow(lbp_img, cmap="nipy_spectral", aspect="auto")
    plt.colorbar(im4b, ax=ax4b, fraction=0.015, pad=0.02)
    ax4b.set_title(f"LBP values — method={lbp_method}, P={lbp_P}, R={lbp_R}")
    ax4b.axis("off")
    st.image(fig_to_pil(fig4b), width='stretch')
    plt.close(fig4b)

    # =========================================================================
    # LBP STEP 5 – Uniform patterns
    # =========================================================================
    st.markdown("---")
    st.header("Step 5 · Uniform Patterns")
    st.markdown(f"""
A **uniform pattern** has **≤ 2 transitions** (0→1 or 1→0) when the circular bit string is read.
They map to interpretable micro-structures:

| Pattern | Example bits | Represents |
|---|---|---|
| All zeros | 00000000 | Flat dark region |
| All ones | 11111111 | Flat bright region |
| One arc of 1s | 00111100 | Edge / ramp |
| Two arcs of 1s | 11000011 | Corner |
| Non-uniform | 01101010 | Complex / noise |

For **P = {lbp_P}**: **{lbp_P*(lbp_P-1)+2} uniform** patterns out of {min(2**lbp_P, 2**lbp_P)} total.
""")
    if lbp_P <= 12 and lbp_method != "var":
        def _is_uniform(code, P):
            bs = [(code >> i) & 1 for i in range(P)]
            return sum(bs[i] != bs[(i + 1) % P] for i in range(P)) <= 2
        uni_codes = [c for c in range(2 ** lbp_P) if _is_uniform(c, lbp_P)]
        st.markdown(
            f"✅ **{len(uni_codes)} uniform** / {2**lbp_P} total "
            f"({100*len(uni_codes)/(2**lbp_P):.1f}%). "
            f"Non-uniform: {2**lbp_P - len(uni_codes)}."
        )
        with st.expander("Show first 12 uniform patterns"):
            show_u = min(12, len(uni_codes))
            fig5u, axs5u = plt.subplots(1, show_u, figsize=(show_u * 1.4, 1.8))
            if show_u == 1:
                axs5u = [axs5u]
            for i, code in enumerate(uni_codes[:show_u]):
                bs = [(code >> b) & 1 for b in range(lbp_P)]
                axs5u[i].bar(range(lbp_P), bs,
                             color=["lime" if b else "tomato" for b in bs], width=0.9)
                axs5u[i].set_title(str(code), fontsize=7)
                axs5u[i].set_xticks([]); axs5u[i].set_yticks([])
            plt.suptitle("Uniform codes — green=bit 1, red=bit 0", fontsize=8)
            plt.tight_layout()
            st.image(fig_to_pil(fig5u), width='stretch')
            plt.close(fig5u)
    else:
        st.info(f"P={lbp_P} → {2**lbp_P} possible codes; pattern enumeration skipped for large P. "
                "The `uniform` / `nri_uniform` methods handle this efficiently internally.")

    # =========================================================================
    # LBP STEP 6 – Histogram (feature descriptor)
    # =========================================================================
    st.markdown("---")
    st.header("Step 6 · LBP Histogram — The Feature Descriptor")
    if lbp_method == "uniform":
        n_bins_lbp     = lbp_P + 2
        hist_range_lbp = (0, lbp_P + 2)
    elif lbp_method == "nri_uniform":
        n_bins_lbp     = lbp_P * (lbp_P - 1) + 3
        hist_range_lbp = (0, lbp_P * (lbp_P - 1) + 3)
    elif lbp_method == "var":
        # var method can produce NaN/Inf on uniform regions; sanitise first
        lbp_img = np.nan_to_num(lbp_img, nan=0.0, posinf=0.0, neginf=0.0)
        n_bins_lbp     = 256
        _vmin = float(np.nanmin(lbp_img))
        _vmax = float(np.nanmax(lbp_img))
        if _vmin == _vmax:
            _vmax = _vmin + 1.0          # avoid degenerate range
        hist_range_lbp = (_vmin, _vmax)
    else:
        n_bins_lbp     = min(256, 2 ** lbp_P)
        hist_range_lbp = (0, min(256, 2 ** lbp_P))
    st.markdown(f"""
The **frequency histogram** of LBP codes over the whole image is the feature vector.
- Method `{lbp_method}` → **{n_bins_lbp} bins** per region
- With a **{grid_y}×{grid_x} spatial grid**: {grid_y * grid_x} cells × {n_bins_lbp} bins
  = **{n_bins_lbp * grid_y * grid_x} total features** per image
""")
    hist_full, _ = np.histogram(lbp_img.ravel(), bins=n_bins_lbp,
                                range=hist_range_lbp, density=True)
    fig6, ax6 = plt.subplots(figsize=(10, 3))
    ax6.bar(range(n_bins_lbp), hist_full, color="steelblue", width=0.85, edgecolor="none")
    ax6.set_xlabel("LBP bin index")
    ax6.set_ylabel("Normalised frequency")
    ax6.set_title(f"Full-image LBP histogram  (method={lbp_method}, P={lbp_P}, R={lbp_R})")
    st.image(fig_to_pil(fig6), width='stretch')
    plt.close(fig6)

    # =========================================================================
    # LBP STEP 7 – Spatial grid descriptor
    # =========================================================================
    st.markdown("---")
    st.header("Step 7 · Spatial LBP Descriptor (Grid of Histograms)")
    st.markdown(
        f"Divide the image into a **{grid_y}×{grid_x}** grid. "
        "Compute one LBP histogram per cell, then concatenate — "
        "this encodes **where** textures appear, not just **which** textures exist."
    )
    cell_h_lbp = img_gray.shape[0] // grid_y
    cell_w_lbp = img_gray.shape[1] // grid_x
    spatial_hists = np.zeros((grid_y, grid_x, n_bins_lbp))
    for gy in range(grid_y):
        for gx in range(grid_x):
            cell_lbp = lbp_img[gy * cell_h_lbp:(gy + 1) * cell_h_lbp,
                                gx * cell_w_lbp:(gx + 1) * cell_w_lbp]
            h, _ = np.histogram(cell_lbp, bins=n_bins_lbp,
                                range=hist_range_lbp, density=True)
            spatial_hists[gy, gx] = h
    dominant_bin  = spatial_hists.argmax(axis=2)
    descriptor_lbp = spatial_hists.reshape(-1)
    fig7, axes7 = plt.subplots(1, 2, figsize=(13, 4))
    im7 = axes7[0].imshow(dominant_bin, cmap="tab20", aspect="auto",
                          vmin=0, vmax=n_bins_lbp - 1)
    axes7[0].set_title(f"Dominant LBP bin per cell ({grid_y}×{grid_x} grid)")
    axes7[0].set_xlabel("Column"); axes7[0].set_ylabel("Row")
    plt.colorbar(im7, ax=axes7[0], fraction=0.046, pad=0.04)
    axes7[1].plot(descriptor_lbp, linewidth=0.7, color="steelblue")
    axes7[1].set_xlabel("Feature index")
    axes7[1].set_ylabel("Value")
    axes7[1].set_title(f"Spatial LBP descriptor (length = {len(descriptor_lbp)})")
    plt.tight_layout()
    st.image(fig_to_pil(fig7), width='stretch')
    plt.close(fig7)

    # ── LBP Summary ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📐 LBP Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Image size",     f"{resize_to}×{resize_to}")
    m2.metric("P (neighbours)", str(lbp_P))
    m3.metric("R (radius)",     str(lbp_R))
    m4.metric("Method",         lbp_method)
    m5.metric("Descriptor len", str(len(descriptor_lbp)))

    st.stop()  # skip HOG pipeline below


# =============================================================================
# STEP 1 – Original & Grayscale
# =============================================================================
st.markdown("---")
st.header("Step 1 · Original Image & Grayscale Conversion")
st.markdown(
    "HOG operates on a **grayscale** image. Colour information is discarded; "
    "only intensity values matter for gradient computation."
)
c1, c2 = st.columns(2)
with c1:
    st.subheader("Original (RGB)")
    st.image(img_resized, caption=f"Resized to {resize_to}×{resize_to}", width='stretch')
with c2:
    st.subheader("Grayscale")
    st.image(img_gray.astype(np.uint8), caption="Grayscale", clamp=True, width='stretch')


# =============================================================================
# STEP 2 – Gradient Computation
# =============================================================================
st.markdown("---")
st.header("Step 2 · Gradient Computation (Gx, Gy, Magnitude, Direction)")
st.markdown(
    """
Gradients capture how intensity **changes** along X and Y axes using the Sobel kernel.
- **Gx** = horizontal gradient (left → right changes)
- **Gy** = vertical gradient (top → bottom changes)
- **Magnitude** $G = \\sqrt{G_x^2 + G_y^2}$ — strength of the edge
- **Direction** $\\theta = \\arctan\\!\\left(\\frac{G_y}{G_x}\\right)$ — orientation of the edge
"""
)

Gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=1)
Gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=1)
mag, ang = cv2.cartToPolar(Gx, Gy, angleInDegrees=True)
ang_unsigned = ang % 180  # unsigned gradient (0–180)

def normalise(arr):
    a = arr - arr.min()
    return (a / (a.max() + 1e-8) * 255).astype(np.uint8)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.image(normalise(Gx),  caption="Gx – Horizontal gradient", clamp=True, width='stretch')
with c2:
    st.image(normalise(Gy),  caption="Gy – Vertical gradient",   clamp=True, width='stretch')
with c3:
    st.image(normalise(mag), caption="Magnitude",                 clamp=True, width='stretch')
with c4:
    # colour-code direction using HSV
    hsv = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = ang_unsigned / 180.0          # hue = direction
    hsv[..., 1] = 1.0                            # full saturation
    hsv[..., 2] = normalise(mag) / 255.0         # value = magnitude
    dir_rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
    st.image(dir_rgb, caption="Direction (hue=angle, brightness=magnitude)", width='stretch')


# =============================================================================
# STEP 3 – Cell Formation
# =============================================================================
st.markdown("---")
st.header("Step 3 · Cell Formation")
ppc = pixels_per_cell
n_cells_x = img_gray.shape[1] // ppc
n_cells_y = img_gray.shape[0] // ppc

st.markdown(
    f"The image is divided into a grid of **{n_cells_x} × {n_cells_y} = {n_cells_x*n_cells_y} cells**, "
    f"each **{ppc}×{ppc} pixels**. A gradient orientation histogram is computed per cell."
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img_gray, cmap="gray")
axes[0].set_title("Grayscale with Cell Grid")
for i in range(n_cells_y + 1):
    axes[0].axhline(i * ppc, color="cyan", linewidth=0.5, alpha=0.7)
for j in range(n_cells_x + 1):
    axes[0].axvline(j * ppc, color="cyan", linewidth=0.5, alpha=0.7)
axes[0].axis("off")

# Show per-cell average gradient magnitude as a heat-map
cell_mag = np.zeros((n_cells_y, n_cells_x))
for cy in range(n_cells_y):
    for cx in range(n_cells_x):
        patch = mag[cy*ppc:(cy+1)*ppc, cx*ppc:(cx+1)*ppc]
        cell_mag[cy, cx] = patch.mean()

im = axes[1].imshow(cell_mag, cmap="hot", aspect="auto")
axes[1].set_title(f"Mean Gradient Magnitude per Cell ({n_cells_y}×{n_cells_x} grid)")
axes[1].set_xlabel("Cell column")
axes[1].set_ylabel("Cell row")
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout()
st.image(fig_to_pil(fig), width='stretch')
plt.close(fig)


# =============================================================================
# STEP 4 – HOG Histogram per Cell
# =============================================================================
st.markdown("---")
st.header("Step 4 · Gradient Orientation Histograms per Cell")
st.markdown(
    f"Each cell accumulates a **{orientations}-bin histogram** of gradient orientations "
    f"(0°–180°, unsigned). Gradient **magnitudes** act as votes – larger edges contribute more."
)

bin_edges   = np.linspace(0, 180, orientations + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Build histogram for every cell
cell_hists = np.zeros((n_cells_y, n_cells_x, orientations))
for cy in range(n_cells_y):
    for cx in range(n_cells_x):
        a_patch = ang_unsigned[cy*ppc:(cy+1)*ppc, cx*ppc:(cx+1)*ppc].ravel()
        m_patch = mag[cy*ppc:(cy+1)*ppc, cx*ppc:(cx+1)*ppc].ravel()
        cell_hists[cy, cx], _ = np.histogram(a_patch, bins=bin_edges, weights=m_patch)

# ── Animated cell walkthrough ────────────────────────────────────────────────
st.markdown("#### 🎞️ Animated Cell Walkthrough")
st.markdown("Each frame highlights one cell, shows its magnitude patch and orientation histogram.")

# cap frames so the GIF stays small
MAX_FRAMES = min(n_cells_y * n_cells_x, 64)
step_y = max(1, n_cells_y * n_cells_x // MAX_FRAMES)
cell_coords = [(ry, rx) for ry in range(n_cells_y) for rx in range(n_cells_x)][::step_y]

fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4))
plt.tight_layout(pad=2)

im_ax   = axes4[0].imshow(img_resized)
grid_lines_h = [axes4[0].axhline(i * ppc, color="cyan", lw=0.4, alpha=0.5) for i in range(n_cells_y + 1)]
grid_lines_v = [axes4[0].axvline(j * ppc, color="cyan", lw=0.4, alpha=0.5) for j in range(n_cells_x + 1)]
cell_rect4 = patches.Rectangle((0, 0), ppc, ppc, lw=2, edgecolor="yellow", facecolor="yellow", alpha=0.4)
axes4[0].add_patch(cell_rect4)
axes4[0].axis("off")

mag_im = axes4[1].imshow(np.zeros((ppc, ppc)), cmap="hot", vmin=0, vmax=mag.max())
axes4[1].axis("off")
axes4[1].set_title("Magnitude patch")

bars4 = axes4[2].bar(bin_centers, np.zeros(orientations), width=180/orientations - 1,
                     color="steelblue", edgecolor="white")
axes4[2].set_xlim(0, 180); axes4[2].set_ylim(0, cell_hists.max() * 1.1 + 1)
axes4[2].set_xlabel("Orientation (°)"); axes4[2].set_ylabel("Weighted vote")
title4 = axes4[0].set_title("Cell (0,0)")

def _update_cell(frame_idx):
    ry, rx = cell_coords[frame_idx % len(cell_coords)]
    cell_rect4.set_xy((rx * ppc, ry * ppc))
    title4.set_text(f"Cell ({ry},{rx})")
    mag_patch = mag[ry*ppc:(ry+1)*ppc, rx*ppc:(rx+1)*ppc]
    mag_im.set_data(mag_patch)
    h = cell_hists[ry, rx]
    for bar, val in zip(bars4, h):
        bar.set_height(val)
    return [cell_rect4, mag_im, title4, *bars4]

anim4 = FuncAnimation(fig4, _update_cell, frames=len(cell_coords), interval=300, blit=False)
tmp4 = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
tmp4.close()  # release handle so Windows lets PillowWriter write & unlink later
anim4.save(tmp4.name, writer=PillowWriter(fps=4))
plt.close(fig4)
with open(tmp4.name, "rb") as f:
    st.image(f.read(), width='stretch')
_os.unlink(tmp4.name)

# ── All cell histograms grid ────────────────────────────────────────────────
max_show = 8
show_y = min(n_cells_y, max_show)
show_x = min(n_cells_x, max_show)
st.markdown(f"#### All Cell Histograms (first {show_y}×{show_x} cells)")
fig, axes = plt.subplots(show_y, show_x, figsize=(show_x * 1.6, show_y * 1.4))
if show_y == 1 and show_x == 1:
    axes = np.array([[axes]])
elif show_y == 1:
    axes = axes[np.newaxis, :]
elif show_x == 1:
    axes = axes[:, np.newaxis]
for ry in range(show_y):
    for rx in range(show_x):
        axes[ry, rx].bar(range(orientations), cell_hists[ry, rx], color="steelblue", width=0.8)
        axes[ry, rx].set_xticks([]); axes[ry, rx].set_yticks([])
        axes[ry, rx].set_title(f"({ry},{rx})", fontsize=6)
plt.suptitle("Per-cell gradient orientation histograms", fontsize=10)
plt.tight_layout()
st.image(fig_to_pil(fig), width='stretch')
plt.close(fig)


# =============================================================================
# STEP 5 – Block Formation & Normalisation
# =============================================================================
st.markdown("---")
st.header("Step 5 · Block Formation & Normalisation")
cpb = cells_per_block
n_blocks_x = n_cells_x - cpb + 1
n_blocks_y = n_cells_y - cpb + 1

st.markdown(
    f"Cells are grouped into overlapping **{cpb}×{cpb}-cell blocks** (stride = 1 cell).  \n"
    f"With {n_cells_y}×{n_cells_x} cells this gives **{n_blocks_y}×{n_blocks_x} = {n_blocks_y*n_blocks_x} blocks**.  \n"
    "Each block descriptor is **L2-Hys normalised** to reduce the effect of illumination changes."
)

# ── Animated block walkthrough ───────────────────────────────────────────────
st.markdown("#### 🎞️ Animated Block Walkthrough")
st.markdown(
    "Each frame sweeps to a new block (red), showing the cell histograms inside it "
    "and the raw vs L2-Hys normalised descriptor."
)

MAX_BFRAMES = min(n_blocks_y * n_blocks_x, 48)
block_coords = [(by, bx) for by in range(n_blocks_y) for bx in range(n_blocks_x)]
step_b = max(1, len(block_coords) // MAX_BFRAMES)
block_coords = block_coords[::step_b]

block_size_px = cpb * ppc
eps = 1e-5
_tab = plt.cm.get_cmap("tab10", cpb * cpb)

fig5 = plt.figure(figsize=(14, 5))
gs = fig5.add_gridspec(2, 2, width_ratios=[1.3, 1])
ax5_img  = fig5.add_subplot(gs[:, 0])
ax5_hist = fig5.add_subplot(gs[0, 1])
ax5_norm = fig5.add_subplot(gs[1, 1])

ax5_img.imshow(img_resized)
for i in range(n_cells_y + 1):
    ax5_img.axhline(i * ppc, color="cyan", lw=0.3, alpha=0.4)
for j in range(n_cells_x + 1):
    ax5_img.axvline(j * ppc, color="cyan", lw=0.3, alpha=0.4)
block_rect5 = patches.Rectangle((0, 0), block_size_px, block_size_px,
                                  lw=2.5, edgecolor="red", facecolor="red", alpha=0.25)
ax5_img.add_patch(block_rect5)
ax5_img.axis("off")
title5 = ax5_img.set_title("Block (0,0)")

# pre-compute max descriptor length for ylim
max_raw = max(
    max(cell_hists[by + dy, bx + dx])
    for by, bx in block_coords
    for dy in range(cpb) for dx in range(cpb)
) * 1.1 + 1

# placeholder bars (cpb² groups × orientations)
n_sub5  = cpb * cpb
x_off5  = np.arange(orientations)
bw5     = 0.8 / n_sub5
bar_groups5 = [
    ax5_hist.bar(x_off5 + k * bw5, np.zeros(orientations), width=bw5,
                 label=f"c{k}", color=_tab(k), alpha=0.85)
    for k in range(n_sub5)
]
ax5_hist.set_ylim(0, max_raw)
ax5_hist.set_xlabel("Orientation bin"); ax5_hist.set_ylabel("Vote")
ax5_hist.legend(fontsize=6, ncol=2)

desc_len5 = n_sub5 * orientations
bars_raw5 = ax5_norm.bar(range(desc_len5), np.zeros(desc_len5), color="steelblue", width=0.8)
bars_nrm5 = ax5_norm.bar(range(desc_len5), np.zeros(desc_len5), color="tomato",   width=0.8, alpha=0.7)
ax5_norm.set_ylim(0, 0.25)
ax5_norm.set_xlabel("Bin index")
ax5_norm.set_title("Raw (blue) vs L2-Hys (red)")

plt.tight_layout(pad=1.5)

def _update_block(frame_idx):
    by, bx = block_coords[frame_idx % len(block_coords)]
    block_rect5.set_xy((bx * ppc, by * ppc))
    title5.set_text(f"Block ({by},{bx})")
    # gather cell hists
    hists_in = [cell_hists[by + dy, bx + dx] for dy in range(cpb) for dx in range(cpb)]
    for k, (bar_grp, hv) in enumerate(zip(bar_groups5, hists_in)):
        for bar, val in zip(bar_grp, hv):
            bar.set_height(val)
    # normalisation
    raw = np.concatenate(hists_in)
    l2  = raw / (np.linalg.norm(raw) + eps)
    clp = np.minimum(l2, 0.2)
    nrm = clp / (np.linalg.norm(clp) + eps)
    for bar, r, n in zip(bars_raw5, raw / (raw.max() + eps) * 0.22, nrm):
        bar.set_height(r); bar.set_height(r)  # scaled raw for visual comparison
    for bar, n in zip(bars_nrm5, nrm):
        bar.set_height(n)
    return [block_rect5, title5, *bars_raw5, *bars_nrm5,
            *(b for grp in bar_groups5 for b in grp)]

anim5 = FuncAnimation(fig5, _update_block, frames=len(block_coords), interval=350, blit=False)
tmp5  = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
tmp5.close()  # release handle so Windows lets PillowWriter write & unlink later
anim5.save(tmp5.name, writer=PillowWriter(fps=3))
plt.close(fig5)
with open(tmp5.name, "rb") as f:
    st.image(f.read(), width='stretch')
_os.unlink(tmp5.name)


# =============================================================================
# STEP 5b – Block Descriptor Stacking Visualisation
# =============================================================================
st.markdown("---")
st.header("Step 5b · Block Descriptor Stacking (Concatenation)")
st.markdown(f"""
After L2-Hys normalisation, each block yields a **{cpb}² × {orientations} = {cpb*cpb*orientations}-d** vector.  
All **{n_blocks_y} × {n_blocks_x} = {n_blocks_y*n_blocks_x}** block descriptors are **concatenated** (stacked)
in **row-major** order to form the final HOG feature vector:

$$\\mathbf{{f}} = [\\underbrace{{\\vec b_{{0,0}}}}_{{ {cpb*cpb*orientations} }}\\;|\\;\\vec b_{{0,1}}\\;|\\;\\cdots\\;|\\;\\vec b_{{{n_blocks_y-1},{n_blocks_x-1}}}]
\\quad\\Rightarrow\\quad \\text{{length}} = {n_blocks_y*n_blocks_x} \\times {cpb*cpb*orientations} = {n_blocks_y*n_blocks_x*cpb*cpb*orientations}$$
""")

# compute all normalised block descriptors
block_descs_all = []
for by in range(n_blocks_y):
    for bx in range(n_blocks_x):
        hists_in = [cell_hists[by + dy, bx + dx] for dy in range(cpb) for dx in range(cpb)]
        raw = np.concatenate(hists_in)
        l2  = raw / (np.linalg.norm(raw) + eps)
        clp = np.minimum(l2, 0.2)
        nrm = clp / (np.linalg.norm(clp) + eps)
        block_descs_all.append(nrm)
full_descriptor = np.concatenate(block_descs_all)
blen = cpb * cpb * orientations  # length of one block descriptor

# ── Animated stacking GIF ────────────────────────────────────────────────────
st.markdown("#### 🎞️ Animated Stacking — watch blocks append to the feature vector")
MAX_STACK_FRAMES = min(n_blocks_y * n_blocks_x, 48)
stack_step = max(1, n_blocks_y * n_blocks_x // MAX_STACK_FRAMES)
stack_indices = list(range(0, n_blocks_y * n_blocks_x, stack_step))

fig_s = plt.figure(figsize=(14, 6))
gs_s  = fig_s.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1])
ax_img_s   = fig_s.add_subplot(gs_s[:, 0])
ax_blk_s   = fig_s.add_subplot(gs_s[0, 1])
ax_stack_s = fig_s.add_subplot(gs_s[1, 1])

# image with grid
ax_img_s.imshow(img_resized)
for i in range(n_cells_y + 1):
    ax_img_s.axhline(i * ppc, color="cyan", lw=0.3, alpha=0.4)
for j in range(n_cells_x + 1):
    ax_img_s.axvline(j * ppc, color="cyan", lw=0.3, alpha=0.4)
block_rect_s = patches.Rectangle((0, 0), block_size_px, block_size_px,
                                  lw=2.5, edgecolor="red", facecolor="red", alpha=0.3)
ax_img_s.add_patch(block_rect_s)
ax_img_s.axis("off")
title_img_s = ax_img_s.set_title("Current block")

# single block descriptor bars
bars_blk_s = ax_blk_s.bar(range(blen), np.zeros(blen), color="tomato", width=0.9)
ax_blk_s.set_ylim(0, 0.3)
ax_blk_s.set_xlabel("Bin within block")
ax_blk_s.set_ylabel("Value")
title_blk_s = ax_blk_s.set_title("Normalised block descriptor")

# growing feature vector
stack_line_s, = ax_stack_s.plot([], [], color="steelblue", lw=0.6)
ax_stack_s.set_xlim(0, len(full_descriptor))
ax_stack_s.set_ylim(0, full_descriptor.max() * 1.15 + 0.01)
ax_stack_s.set_xlabel("Feature vector index")
ax_stack_s.set_ylabel("Value")
title_stack_s = ax_stack_s.set_title("Stacked descriptor: 0 / 0 bins")

# vertical separator lines (one per block boundary)
for k in range(1, n_blocks_y * n_blocks_x):
    ax_stack_s.axvline(k * blen, color="gray", lw=0.3, alpha=0.3)

plt.tight_layout(pad=1.5)

def _update_stack(fi):
    bidx = stack_indices[fi % len(stack_indices)]
    by = bidx // n_blocks_x
    bx = bidx % n_blocks_x
    block_rect_s.set_xy((bx * ppc, by * ppc))
    title_img_s.set_text(f"Block ({by},{bx})  —  #{bidx+1} of {n_blocks_y*n_blocks_x}")
    # update single block bars
    d = block_descs_all[bidx]
    for bar, v in zip(bars_blk_s, d):
        bar.set_height(v)
    title_blk_s.set_text(f"Block ({by},{bx}) — {blen} bins")
    # update growing feature vector (show everything up to and including this block)
    end = (bidx + 1) * blen
    stack_line_s.set_data(np.arange(end), full_descriptor[:end])
    title_stack_s.set_text(f"Stacked descriptor: {end} / {len(full_descriptor)} bins")
    # highlight current block region on the feature vector
    return [block_rect_s, title_img_s, title_blk_s, *bars_blk_s, stack_line_s, title_stack_s]

anim_s = FuncAnimation(fig_s, _update_stack, frames=len(stack_indices), interval=400, blit=False)
tmp_s  = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
tmp_s.close()
anim_s.save(tmp_s.name, writer=PillowWriter(fps=3))
plt.close(fig_s)
with open(tmp_s.name, "rb") as f:
    st.image(f.read(), width='stretch')
_os.unlink(tmp_s.name)

st.markdown(f"""
**Stacking summary:**
- Each block → **{blen}** values after L2-Hys normalisation
- {n_blocks_y}×{n_blocks_x} = **{n_blocks_y*n_blocks_x}** blocks processed in row-major order
- Final HOG vector = **{len(full_descriptor)}** elements
""")


# =============================================================================
# STEP 6 – Final HOG Descriptor Visualisation
# =============================================================================
st.markdown("---")
st.header("Step 6 · Final HOG Descriptor Visualisation")
st.markdown(
    "Using **scikit-image** to compute the full HOG descriptor.  "
    "Lines in each cell show the **dominant gradient direction** — "
    "length encodes magnitude.  The final feature vector length is also shown."
)

fd, hog_image = hog(
    img_gray,
    orientations=orientations,
    pixels_per_cell=(ppc, ppc),
    cells_per_block=(cpb, cpb),
    block_norm="L2-Hys",
    visualize=True,
    feature_vector=True,
)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.4))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img_resized)
axes[0].set_title("Original Image")
axes[0].axis("off")
axes[1].imshow(hog_image_rescaled, cmap="gray")
axes[1].set_title("HOG Visualisation")
axes[1].axis("off")
plt.tight_layout()
st.image(fig_to_pil(fig), width='stretch')
plt.close(fig)

st.success(
    f"✅ HOG descriptor computed!  "
    f"Feature vector length = **{len(fd)}**  "
    f"({n_blocks_y} × {n_blocks_x} blocks × {cpb}² cells/block × {orientations} bins)"
)

# ── Feature vector histogram ────────────────────────────────────────────────
with st.expander("📊 Show HOG feature vector distribution"):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(fd, linewidth=0.5, color="steelblue")
    ax.set_xlabel("Feature vector index")
    ax.set_ylabel("Value")
    ax.set_title("Full HOG feature vector")
    st.image(fig_to_pil(fig), width='stretch')
    plt.close(fig)

# ── Summary metrics ─────────────────────────────────────────────────────────
st.markdown("---")
st.header("📐 Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Image size",      f"{resize_to}×{resize_to}")
col2.metric("Cells",           f"{n_cells_y}×{n_cells_x} = {n_cells_y*n_cells_x}")
col3.metric("Blocks",          f"{n_blocks_y}×{n_blocks_x} = {n_blocks_y*n_blocks_x}")
col4.metric("Bins / cell",     f"{orientations}")
col5.metric("Feature vec len", f"{len(fd)}")
