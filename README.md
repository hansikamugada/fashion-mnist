# Fashion-MNIST Image Classification Using CNN

**College Deep Learning Project**  
*Fashion-MNIST Image Classification Using Convolutional Neural Network (CNN)*

---

## 🌟 Overview & Website Link

- **Live Local Website URL:** [http://127.0.0.1:5000](http://127.0.0.1:5000) or [http://localhost:5000](http://localhost:5000)
- **Status:** Trained, Evaluated, and Running Locally
- **Trained Model File:** `fashion_mnist_cnn.keras` (1.69 MB)
- **Overall Test Accuracy:** **91.48%** (evaluated across 10,000 unseen test images)
- **Validation Accuracy:** **92.15%**

---

## 🎨 Features & Implementation

1. **Interactive Image Classifier:**
   - **Upload File:** Drag-and-drop or browse PNG, JPG, JPEG, and WEBP images.
   - **Camera Capture:** Live browser webcam/camera capture with framing guides, snap button, and retake controls.
   - **Quick Benchmark Samples:** Clickable gallery of 10 test images (1 per class) for instant review testing.
   - **Smart Background Inversion:** Automatically detects light camera/photo backgrounds and inverts pixel values to match Fashion-MNIST distribution (dark background, bright foreground).
   - **Model Tensor Inspection:** Live side-by-side view showing the preprocessed $28 \times 28$ grayscale tensor as fed to the CNN.
   - **Confidence & Probabilities:** Prominent prediction badge, circular confidence meter, and sorted 10-class probability distribution bars.

2. **10 Supported Fashion-MNIST Categories:**
   - Class 0: T-shirt / Top
   - Class 1: Trouser
   - Class 2: Pullover
   - Class 3: Dress
   - Class 4: Coat
   - Class 5: Sandal
   - Class 6: Shirt
   - Class 7: Sneaker
   - Class 8: Bag
   - Class 9: Ankle Boot

3. **Academic Evaluation & Results:**
   - Training & Validation Accuracy and Loss Curves (`static/images/accuracy_loss.png`)
   - Test Confusion Matrix Heatmap (`static/images/confusion_matrix.png`)
   - Complete Classification Report Table with Precision, Recall, and F1-Score per class.

---

## 🧠 CNN Architecture

```
Input (28 × 28 × 1 Grayscale)
  │
  ├── Conv2D (32 filters, 3×3 kernel, ReLU, 'same' padding)
  ├── MaxPooling2D (2×2 pool size) ──> Output: (14 × 14 × 32)
  │
  ├── Conv2D (64 filters, 3×3 kernel, ReLU, 'same' padding)
  ├── MaxPooling2D (2×2 pool size) ──> Output: (7 × 7 × 64)
  │
  ├── Flatten ──> 3,136 features
  ├── Dense (128 units, ReLU)
  ├── Dropout (0.3 rate)
  └── Dense (10 units, Softmax activation)
```

---

## 📁 Project File Structure

```
FASION FINST/
├── app.py                     # Flask REST backend server
├── train_model.py             # CNN training & evaluation script
├── fashion_mnist_cnn.keras    # Saved trained CNN model
├── fashion_mnist_cnn.pt       # PyTorch checkpoint backup
├── templates/
│   └── index.html             # Main responsive HTML interface
├── static/
│   ├── css/
│   │   └── style.css          # Modern fashion-tech UI stylesheet
│   ├── js/
│   │   └── main.js            # Image upload, camera controller & charts
│   ├── data/
│   │   ├── metrics.json       # Empirical test evaluation metrics
│   │   └── sample_catalog.json# Test samples catalog
│   └── images/
│       ├── accuracy_loss.png  # Training curves plot
│       ├── confusion_matrix.png # Confusion matrix plot
│       └── samples/           # 10 benchmark test sample PNGs
└── README.md                  # Project documentation
```

---

## 🚀 Running the Project

To restart or run the server locally:

```bash
python app.py
```

Then open your browser at:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**
