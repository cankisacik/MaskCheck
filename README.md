
# MaskCheck: Real-Time Face Mask Detection Using CNN-Transformer Hybrid

## Overview
MaskCheck is a deep learning-based face mask detection system combining Convolutional Neural Networks (CNNs) with transformer-inspired mechanisms. It classifies face mask usage into:

- **Mask Worn Properly**
- **No Mask**
- **Mask Worn Incorrectly**

The model integrates advanced attention modules and interpretable AI features like Grad-CAM to offer insight into its decision-making.

---

## Project Structure
```
├── maskcheck_final_model.keras          # Final saved model
├── maskcheck_best_model.keras           # Best model (early stopping)
├── maskcheck_training_history.pkl       # Training metrics
├── cam.jpg                              # Example Grad-CAM result
├── archive.zip                          # Dataset (Kaggle)
├── MaskCheckPlus.ipynb                  # Full model and training notebook
└── README_maskcheck_updated.md          # This file
```

---

## Installation & Setup

### 1. Dataset
- Source: [Face Mask Detection Dataset by Ashish Jangra on Kaggle](https://www.kaggle.com/datasets/ashishjangra/facemask-detection)
- Upload `archive.zip` to your Colab environment and extract it.

### 2. Requirements
```bash
pip install tensorflow opencv-python matplotlib scikit-learn pandas
```

---

## Model Training

Run the code in `MaskCheck.py`.

Model components:
- Frozen **MobileNetV2** as backbone
- Attention modules:
  - **CBAM (Convolutional Block Attention Module)**
  - **ECA (Efficient Channel Attention)**
  - **SE (Squeeze-and-Excitation)**
  - **Non-local Block**
  - **Multi-head Self-Attention (MHSA)**
- Custom convolutional residual block
- Optimizer: Adam, LR: 0.0001, Epochs: 10, Batch Size: 32

---

## Evaluation

| Metric         | Value    |
|----------------|----------|
| Accuracy       | ~84.8%   |
| Validation Loss| ~0.53    |

Confusion Matrix and classification report are available via `sklearn.metrics`.

---

## Grad-CAM Visualization

To interpret model predictions:
```python
from google.colab import files
uploaded = files.upload()
predict_mask_status("your_image.jpg")
```
You will see the predicted class and a Grad-CAM heatmap.

---

## Model Loading with Custom Layers

Since custom layers like `ReduceMean`, `ReduceMax`, `ExpandDimsTwice` are used, load the model as follows:

```python
model = load_model("maskcheck_final_model.keras", custom_objects={
    'ReduceMean': ReduceMean,
    'ReduceMax': ReduceMax,
    'ExpandDimsTwice': ExpandDimsTwice
})
```

---

## Future Improvements

- Dataset diversity can be extended for better generalization
- CNN backbone can be replaced with a pure Vision Transformer (ViT)
- A web-cam can be integrated for real-time mask detection

---

## Author

**Can Kısacık**, Bahçeşehir University  
Department of Artificial Intelligence Engineering  
2025
