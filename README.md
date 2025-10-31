
# 🌾 RiML – Rice Stress Predictor

**RiML (Rice Machine Learning Predictor)** is a Streamlit-based web application designed to analyze gene expression data in rice and predict **stress conditions** such as *Control, Drought, Salt,* and *Cold*.  
The tool leverages a trained Machine Learning model to support research in **plant genomics** and **agricultural stress analysis**.

---

## 🧠 Overview

Rice is one of the world’s most important staple crops, but its yield is affected by various abiotic stress factors.  
Using a pre-trained ML model, **RiML** helps identify which stress condition a rice sample is undergoing based on its gene expression profile.

The app allows users to:
- Upload gene expression data (matrix file in `.txt`, `.csv`, `.tsv`, or `.xlsx` format)
- Optionally upload a **GPL annotation file** for probe-to-gene mapping
- Automatically preprocess and match genes with the model’s features
- Run predictions and visualize results interactively

---

## 🚀 Features

- 🧬 **Model-Based Prediction:** Detects rice stress type using a pre-trained ML classifier  
- 📊 **Interactive Analysis Tab:** Upload, preprocess, and visualize your data  
- 📈 **Graphical Outputs:** Bar and pie charts showing stress distribution  
- 🏠 **Home Tab:** Overview and introduction cards for easy navigation  
- ℹ️ **About Tab:** Detailed project insights, future trends, acknowledgements, and team info  
- 💾 **Download Results:** Export predictions as a `.csv` file  

---

## 📈 Model Performance

After preprocessing and feature selection, **40,080 genes** were filtered, and **165 significant stress-responsive genes** were identified (with **341 genes having adjusted p < 0.05**).

---

### 🔍 Evaluation Summary
The trained ML model was evaluated using **Leave-One-Out Cross-Validation (LOOCV)** to ensure unbiased performance on small sample sizes.

| Metric | Score |
|:--------|:-------|
| **Accuracy** | 0.833 |
| **Macro F1-score** | 0.837 |
| **Weighted F1-score** | 0.838 |

---

### 🧩 Classification Report
| Class | Precision | Recall | F1-score | Support |
|:------|:-----------|:--------|:----------|:----------|
| Cold | 1.00 | 0.67 | 0.80 | 3 |
| Control | 1.00 | 1.00 | 1.00 | 3 |
| Drought | 0.60 | 1.00 | 0.75 | 3 |
| Salt | 1.00 | 0.67 | 0.80 | 3 |

**Accuracy:** 0.83  
**Macro avg:** Precision = 0.90 | Recall = 0.83 | F1 = 0.84  
**Weighted avg:** Precision = 0.90 | Recall = 0.83 | F1 = 0.84  

---

### 🌾 Per-Stress Type Accuracy
| Stress Type | Accuracy | Samples |
|--------------|-----------|----------|
| Control | 100.0% | 3/3 |
| Drought | 100.0% | 3/3 |
| Salt | 66.7% | 2/3 |
| Cold | 66.7% | 2/3 |

---

### 🧠 Interpretation
- The model achieved **~83% accuracy** overall, showing strong generalization for small sample sizes.  
- It performs perfectly for **Control** and **Drought** stress conditions.  
- **Cold** and **Salt** stresses show moderate classification, suggesting room for improvement with additional training data.  
- These results validate the model’s capability to differentiate between stress conditions based on rice gene expression profiles.


## 🧩 Tech Stack

- **Language:** Python  
- **Framework:** Streamlit  
- **Libraries:** pandas, numpy, joblib, pickle, matplotlib  
- **Model Type:** Machine Learning classifier trained on rice gene expression data  

---

## ⚙️ Installation and Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/codepranjal829/Plant-stress-tolerance-ML-model-Rice-.git

cd RiML-RiceStressPredictor

# Future Trends
Integration with deep learning for multi-stress prediction
Expansion to other crops such as maize, barley, and wheat
Enhanced interpretability with gene importance dashboards
