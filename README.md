# 🫁 Pulmonary Disease Diagnosis from CT Scans  
### End-to-End Hybrid ML-DL & MLOps System

---

## 🧩 System Architecture

> Architecture.png

---

## 🎥 Demo

---


---

## 🚀 Overview

This project implements a **production-grade hybrid ML+DL pipeline** for pulmonary disease classification from CT scan images.

The system achieves **>96% Macro F1-score** on the evaluation dataset, demonstrating strong balanced classification performance across multiple pulmonary conditions.

Unlike typical academic implementations, this project incorporates a **full MLOps lifecycle**, including:

- Data pipelines  
- Experiment tracking  
- Model versioning  
- CI/CD workflows  
- Cloud-ready deployment  

---

## 🧠 Key Highlights

- ✅ **>96% F1-score (macro)**
- ✅ End-to-end ML pipeline (**data → training → deployment**)
- ✅ Hybrid CNN + ML classifier approach
- ✅ Experiment tracking & reproducibility
- ✅ CI/CD pipeline (production-ready)
- ✅ Real-time inference API + web UI

---

## 🛠️ Technologies Used

### 🧑‍💻 Languages
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

### 🔄 Version Control & Data Versioning
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=data-version-control&logoColor=white)

---

### 🤖 Machine Learning & Frameworks
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)

---

### ⚙️ Backend & Deployment
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white)

---

### 🗄️ Data Source
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

---

## 📊 Model Performance

| Metric        | Score |
|--------------|------|
| **F1-score** | **> 96%** |
| Precision    | High |
| Recall       | High |

> > ⚠️ **Evaluation Note**
>
> Reported performance (**F1 > 96%**) is measured on a held-out test dataset.
> To ensure validity:
> - Data is split into **train/test sets with no overlap**
> - Evaluation is performed on **unseen test data only**
>
> ⚠️ Medical imaging models are sensitive to data leakage and dataset bias. 
> Further validation (e.g., patient-level split, cross-dataset evaluation) is recommended for clinical generalization.

---

## 📥 Installation & Usage

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Shah-xai/HealthCare-Pulmonary-diagnosis.git
cd HealthCare-Pulmonary-diagnosis
pip install -r requirements.txt
dvc pull
dvc repro
python app.py
http://localhost:5000
```
## 📚 Reference

This project is inspired by the following research work:

- *“A hybrid deep learning and machine learning approach for lung cancer classification using CT images,”*  
  **Expert Systems with Applications**, 2023.  
  https://www.sciencedirect.com/science/article/pii/S0957417423004633  

Key contributions adopted in this project:

- Use of **CNN-based feature extraction from CT images**
- Integration of **dimensionality reduction techniques (e.g., PCA)**
- Application of **classical ML classifiers (e.g., SVM)** on deep features
- Demonstration of improved classification performance using hybrid pipelines :contentReference[oaicite:0]{index=0}
