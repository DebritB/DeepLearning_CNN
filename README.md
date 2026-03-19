# DeepLearning CNN Projects

A multi-project repository for deep learning and computer vision projects with interactive visualizations.

## Projects

### 4. **CNN** - Convolutional Neural Network Visualizer
An interactive Streamlit application for understanding CNN operations with stunning Manim animations. Features a dark-themed UI with comprehensive step-by-step visualizations of every CNN layer.

**Features:**

*Interactive Streamlit App:*
- ⚙️ **Convolution Configuration** - Customize input size, kernel size, padding, stride, and animation speed
- 📊 **Convolution Process** - View kernel matrices with heatmap coloring and output dimension formulas
- 🎬 **Kernel Animation** - Watch the kernel slide across the input with shadow highlighting and perspective lines
- 🏊 **Pooling Operations** - Visualize both Max Pool and Average Pool with configurable pool size and stride
- 🔄 **Flatten Layer** - 2D to 1D transformation with animated flying dots
- 🖼️ **Full Pipeline Demo** - Upload your own image and watch it flow through the complete CNN
- 🎥 **Manim Video Generation** - Professional animated videos rendered with Manim

*Manim Animations (`cnn_artistic.py`):*
- 📽️ **Complete CNN Pipeline** - Input → Convolution → Pooling → Flatten → Dense → Output
- 📚 **3D Stacked Feature Maps** - Visualize 6 filter layers with depth perspective effect
- 🔙 **Backpropagation Animation** - Error gradient flowing backward, updating weights layer by layer
- 🎯 **Training Loop Demo** - Wrong prediction → Backprop → Correct prediction workflow
- ✨ **Custom Image Support** - Uses your uploaded image in the animation

**Location:** `CNN/`
**Tech Stack:** Streamlit, Manim, NumPy, PIL, Matplotlib, Pandas

**To Run:**
```bash
cd CNN
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8503
```

### 2. **HOG_LBP** - Feature Descriptor Visualizer
A Streamlit application for visualizing and exploring HOG (Histogram of Oriented Gradients) and LBP (Local Binary Pattern) feature descriptors.

**Location:** `HOG_LBP/`
**Tech Stack:** Streamlit, OpenCV, scikit-image, NumPy, Matplotlib

**To Run:**
```bash
cd HOG_LBP
python -m streamlit run app.py --server.port 8501
```

### 3. **Neural Network** - Deep Learning Fundamentals Visualizer
An interactive Streamlit application for understanding neural network concepts with visual animations and step-by-step walkthroughs.

**Covers:**
- 🔄 Forward Propagation - layer-by-layer computation
- ⚡ Activation Functions - ReLU, Sigmoid, Tanh, Leaky ReLU, ELU
- 📉 Loss Calculation - MSE, Cross Entropy, MAE
- 🔙 Backpropagation - computing gradients through the network
- 🎯 Gradient Descent - SGD, Mini-Batch GD, Batch GD optimization strategies

**Location:** `neural_network/`
**Tech Stack:** Streamlit, NumPy, Matplotlib, scikit-learn

**To Run:**
```bash
cd neural_network
python -m streamlit run app.py --server.port 8502
```

## Repository Structure

```
DLCNNQUIZ/
├── CNN/
│   ├── app.py              # Main Streamlit app
│   ├── cnn_artistic.py     # Manim animation with backpropagation
│   └── requirements.txt
├── HOG_LBP/
│   ├── app.py
│   ├── requirements.txt
│   └── openapp.txt
├── neural_network/
│   ├── app.py
│   └── requirements.txt
└── README.md
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DebritB/DeepLearning_CNN.git
   cd DeepLearning_CNN
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows
   source .venv/bin/activate     # Linux/Mac
   ```

3. **Install dependencies for a specific project:**
   ```bash
   cd CNN
   pip install -r requirements.txt
   ```

## Screenshots

### CNN Animation Pipeline
The CNN visualizer creates professional Manim animations showing:
- Your uploaded image being processed
- Convolution with 3D stacked feature maps
- Max pooling extracting values
- Flatten operation collapsing 3D to 1D
- **Backpropagation** with weight updates visualized
- Training loop: Wrong → Learn → Correct!

---

**Author:** DebritB  
**GitHub:** https://github.com/DebritB/DeepLearning_CNN  
**LinkedIn:** https://www.linkedin.com/in/debrit-bhattacharyya-77622a210/
