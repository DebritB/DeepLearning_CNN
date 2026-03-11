# DeepLearning CNN Projects

A multi-project repository for deep learning and computer vision projects.

## Projects

### 1. **HOG_LBP** - Feature Descriptor Visualizer
A Streamlit application for visualizing and exploring HOG (Histogram of Oriented Gradients) and LBP (Local Binary Pattern) feature descriptors.

**Location:** `HOG_LBP/`
**Tech Stack:** Streamlit, OpenCV, scikit-image, NumPy, Matplotlib

**To Run:**
```bash
cd HOG_LBP
python -m streamlit run app.py --server.port 8501
```

### 2. **Neural Network** - Deep Learning Fundamentals Visualizer
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
├── HOG_LBP/
│   ├── app.py
│   ├── requirements.txt
│   └── openapp.txt
├── [Other Projects]/
└── README.md
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DebritB/DeepLearning_CNN.git
   cd DeepLearning_CNN
   ```

2. **Create a virtual environment (optional, create per project):**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows
   ```

3. **Install dependencies for a specific project:**
   ```bash
   cd HOG_LBP
   pip install -r requirements.txt
   ```

---

**Author:** DebritB  
**GitHub:** https://github.com/DebritB/DeepLearning_CNN
