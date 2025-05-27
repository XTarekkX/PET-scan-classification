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
> 📦 **Weights Resulted**:
> https://drive.google.com/drive/folders/15A0v1cORPAyfY6hsy7XJYqxlDKh17QCL?usp=drive_link

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

> ⚠️ **Disclaimer**:
> - The annotations and labeling were **not fully performed by licensed medical professionals (some are)**.
> - This dataset is intended for **academic and research purposes only**.
> - Any clinical usage, redistribution, or commercial use is **strictly prohibited**.

---

## ⚙️ Training Configuration

### 📌 Loss Function – Focal Loss

This project uses **Focal Loss** instead of standard categorical cross-entropy to address **class imbalance**—a common challenge in medical imaging.

**Why Focal Loss?**

- In imbalanced datasets, standard loss functions can be dominated by frequent classes.
- **Focal Loss** down-weights easy examples and focuses more on hard, misclassified cases.

**Formula**:

> `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`

- **α (alpha)** balances the importance of each class  
  → Here: `[0.5, 0.3, 0.2]` for `No Cancer`, `Breast Cancer`, `Prostate Cancer`
- **γ (gamma)** controls the focus on hard examples  
  → Here: `γ = 2.0` (default in many implementations)

This encourages the model to pay more attention to underrepresented or harder-to-classify classes, such as `Breast Cancer`.

---

### 🔁 Learning Rate Scheduler – Warmup & Decay

A **custom learning rate schedule** is used to stabilize training and prevent convergence to poor minima early on.

#### 🔹 Warmup Phase

- **Epochs 1–5**: Learning rate = `5e-5`  
  → A small but stable rate to “warm up” the model

#### 🔹 Main Phase

- **Epochs 6–14**: Learning rate increases to `1e-4`  
  → Allows faster learning after warmup

- **Epochs 15–24**: Decay begins, reducing LR back to `5e-5`

- **Epochs 25+**: Final decay stage with LR = `1e-5`  
  → Stabilizes convergence and prevents overshooting

Combined with **ReduceLROnPlateau** and **EarlyStopping**, this ensures adaptive and controlled learning across 50 epochs.

---

### 🏋️ Class Weights

To handle **class imbalance**, weights are computed using `sklearn.utils.class_weight`:

| Class            | Weight  |
|------------------|---------|
| No Cancer        | 0.9379  |
| Breast Cancer    | 1.0731  |
| Prostate Cancer  | 0.9981  |

These weights are integrated into the training process to emphasize underrepresented classes during backpropagation.

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

## 📷 Results

### 📈 Accuracy Over Epochs

![Accuracy Plot](https://github.com/XTarekkX/PET-scan-classification/raw/main/results/accuracy%20plot.png)

### 📉 Confusion Matrix

![Confusion Matrix](https://github.com/XTarekkX/PET-scan-classification/raw/main/results/cm%20matrix.png)

### 📊 ROC Curve

![ROC Curve](https://github.com/XTarekkX/PET-scan-classification/raw/main/results/roc%20curve%5D.png)


---

## 📁 Project Structure

📁 project-root/

┣ 📁 results/ ← Visual outputs (plots & metrics)

┃ ┣ 📷 accuracy plot.png

┃ ┣ 📷 cm matrix.png

┃ ┗ 📷 roc curve].png

┣ 📄 README.md ← Project documentation

┗ 📓 inception-over-resnet-80.ipynb ← Training, evaluation, and results


---

## 📌 Future Improvements

- Incorporate additional cancer types or multimodal data (e.g., PET + CT)
- Train with expert-annotated datasets to improve medical reliability
- Use model explainability tools like **Grad-CAM** for visual justification
- Package into a deployable clinical decision support tool

---

## 📜 License

This project is intended for **educational and research purposes only**.  
Use of this dataset or model for clinical diagnosis, treatment, or commercial applications is **strictly prohibited**.

