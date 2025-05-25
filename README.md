# 🔬 Cancer Classification from PET Scans using InceptionResNetV2

This project focuses on the **classification of cancer types** from 2D PET scan images using a deep learning model based on **InceptionResNetV2**, pretrained on **RadImageNet**—a medical imaging alternative to ImageNet.

The objective is to automatically categorize PET scans into one of the following classes:

- **No Cancer**
- **Breast Cancer**
- **Prostate Cancer**

---

## 🧠 Model Architecture

The model is built on the **InceptionResNetV2** architecture with **RadImageNet weights**, specifically optimized for medical imaging tasks.

> 📖 **RadImageNet GitHub Repository**:  
> https://github.com/BMEII-AI/RadImageNet  
> 📦 **Weights Used (Kaggle Dataset)**:  
> https://www.kaggle.com/datasets/ipythonx/notop-wg-radimagenet/data

### Model Summary

| Layer                            | Output Shape         | Parameters     |
|----------------------------------|----------------------|----------------|
| Input                            | (224, 224, 3)        | 0              |
| InceptionResNetV2 (frozen)       | (5, 5, 1536)         | 54,336,736     |
| Global Average Pooling           | (1536)               | 0              |
| Dense (ReLU, 1024 units)         | (1024)               | 1,573,888      |
| Dropout (rate=0.7)               | (1024)               | 0              |
| Dense (Softmax, 3 units)         | (3)                  | 3,075          |
| **Total Parameters**             |                      | **55,913,699** |
| **Trainable Parameters**         |                      | **1,576,963**  |

---

## 🧬 Dataset Information

- **Total Training Images**: `1072`
- **Validation Images**: `58`
- **Test Images**: `63`
- **Image Size**: `(224, 224, 3)`

> 🏥 The dataset was **manually collected** from a hospital in **Egypt**, consisting of real-life PET scan images.

> ✍️ All labels were **manually annotated** by the research team using medical knowledge and tools.

> ⚠️ **Disclaimer**:
> - The annotations were **not fully performed by licensed medical professionals**.
> - This dataset is intended for **academic and research purposes only**.
> - Any clinical usage, redistribution, or commercial use is **prohibited**.

---

## ⚙️ Training Configuration

- **Loss Function**: Weighted Categorical **Focal Loss**  
  > Custom α = `[0.5, 0.3, 0.2]`, γ = `2.0`
- **Optimizer**: Adam
- **Scheduler**: Custom Learning Rate Scheduler  
  → Warmup + Reduce on Plateau
- **Epochs**: 50  
- **Early Stopping**: Patience = 5  
- **Dropout**: 0.7
- **Class Weights** (computed to handle imbalance):
  - `No Cancer`: 0.9379
  - `Breast Cancer`: 1.0731
  - `Prostate Cancer`: 0.9981

---

## 📈 Model Performance

### ✅ Final Evaluation (Test Set)

- **Test Loss**: `0.0839`
- **Test Accuracy**: `0.8095`
- **Overall F1 Score**: `0.8055`

### 📊 Classification Report:

| Class            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| No Cancer        | 0.67      | 0.91   | 0.77     | 22      |
| Breast Cancer    | 0.85      | 0.55   | 0.67     | 20      |
| Prostate Cancer  | 1.00      | 0.95   | 0.98     | 21      |
| **Overall**      |           |        |          | **63**  |
| **Accuracy**     |           |        | **0.81** |         |

---

## 🧪 Training History Snapshot

- Best validation accuracy: **0.7414**
- Early stopping kicked in at epoch 50
- Model saved as: `frozen_model.h5`

---

## 📁 Project Structure

