# 🚀 Higgs Boson Signal Classification — Deep Learning Project

This project applies deep learning techniques to classify Higgs Boson particle collision events as either **signal** (actual Higgs boson) or **background noise** using tabular physics data. The dataset is based on the Kaggle **Higgs Boson Machine Learning Challenge**.

---

## 📁 Dataset

- **Source**: [Kaggle Higgs Boson Challenge](https://www.kaggle.com/competitions/higgs-boson)
- **Files Used**:
  - `higgs_train_10k.csv` – 92,638 samples
  - `higgs_test_5k.csv` – 87,195 samples
- **Target Column**: `Prediction` (binary: 1 = signal, 0 = background)
- **Features**: 30 real-valued, anonymized physics features

---

## 🎯 Objective

Design, train, and evaluate a deep learning model to perform **binary classification**. Tasks include:

- Data preprocessing & normalization
- Model architecture design
- Optimization & regularization
- Evaluation with advanced metrics
- Comparison with a baseline XGBoost model

---

## 🧪 Model Workflow

### 1. Data Preparation

- Handled zero-filled and missing values
- Normalized continuous features using `StandardScaler`
- Split data into:
  - 70% training
  - 15% validation
  - 15% test set

---

### 2. Model Architecture (Keras/TensorFlow)

- **Input Layer**: 30 features
- **Hidden Layers**: Tried various combinations (128-64, 256-128-64)
- **Activations**: `ReLU`, `SELU`
- **Regularization**: `Dropout`, `BatchNormalization`
- **Output Layer**: 1 unit with `sigmoid`
- **Loss**: `binary_crossentropy`
- **Optimizers Tried**:
  - Adam
  - RMSprop
  - AdamW
- **Callbacks**:
  - `EarlyStopping`
  - `ReduceLROnPlateau`

---

### 3. Evaluation Metrics

- Accuracy
- Precision, Recall, F1 Score
- ROC-AUC Curve
- Confusion Matrix
- Training/Validation Accuracy & Loss Curves

---

### 4. Baseline Comparison

A tree-based model using **XGBoost** was implemented for benchmarking:

- `XGBClassifier` with `n_estimators=100`, `max_depth=6`
- Evaluated on the same test set
- Reported Accuracy & ROC-AUC

---

## 🔍 Reflections

- **Model Depth**: Deeper architectures with regularization improved learning without overfitting.
- **Overfitting Mitigation**: Dropout + EarlyStopping + BatchNorm helped stabilize performance.
- **Learning Rate**: Lower values (e.g., `0.0005`) led to better convergence, especially with RMSprop.
- **Future Improvements**:
  - Feature engineering (e.g., interaction terms)
  - Hyperparameter tuning using Optuna/Hyperband
  - Ensembling with tree-based models

---

## 📎 Files Included

- `higgs_train_10k.csv`  
- `higgs_test_5k.csv`  
- `model_training.ipynb`  
- `README.md`

---

## ✅ Requirements

- Python 3.7+
- TensorFlow / Keras
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- XGBoost

---

## 📌 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook model_training.ipynb
