# 🩺 Chronic Kidney Disease Prediction

This project demonstrates a **complete machine learning pipeline** to predict **Chronic Kidney Disease (CKD)** from patient clinical and laboratory data. It includes **data preprocessing, balancing, model training, evaluation, saving artifacts, and deployment as a Streamlit app**.

👉 **Live App:** [Streamlit CKD Predictor](https://chronic-kidney-disease-ckd-predictor---end-to-end-ml-project-m.streamlit.app/)

---

## 📂 Project Structure

```
.
├── kidney_disease.csv          # Dataset
├── CKD_Prediction.ipynb        # Jupyter Notebook (data prep, EDA, model training)
├── app.py                      # Streamlit web app for deployment
├── models/
│   ├── scaler.pkl              # Saved MinMaxScaler for preprocessing
│   └── xgb_model.pkl           # Trained XGBoost model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📊 Dataset

The dataset `kidney_disease.csv` contains patient features and labels for CKD classification.

### Features used:

* **Numerical:** `age`, `bp`, `sg`, `al`, `hemo`, `sc`
* **Categorical:** `htn`, `dm`, `cad`, `appet`, `pc`
* **Target:** `classification` (1 = CKD, 0 = Not CKD)

---

## ⚙️ Workflow

### 1. Data Preprocessing (in Jupyter Notebook)

* Loaded and explored dataset (`df.info()`, `.describe()`).
* Selected **relevant features** from raw dataset.
* Handled missing values:

  * Numerical → median imputation.
  * Categorical → mode imputation.
* Cleaned categorical inconsistencies (e.g., trimming whitespaces).
* Encoded categorical variables into numeric (`yes/no`, `good/poor`, etc.).
* Normalized numerical values using **MinMaxScaler**.

### 2. Handling Class Imbalance

* Observed dataset imbalance (more CKD than non-CKD).
* Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance classes.

### 3. Model Training & Evaluation

* Trained multiple ML models:

  * Logistic Regression
  * Random Forest Classifier
  * Gradient Boosting Classifier
  * K-Nearest Neighbors
  * Support Vector Machine
  * AdaBoost Classifier
  * Gaussian Naive Bayes
  * **XGBoost Classifier (best performing)**

* Metrics used:

  * Accuracy
  * Confusion Matrix
  * Classification Report

✅ **XGBoost Classifier** gave the highest accuracy and was chosen as the final model.

### 4. Saving Artifacts

* Saved trained **scaler** and **XGBoost model** in `models/` folder.

### 5. Deployment (Streamlit App)

* Built `app.py` to:

  * Accept **single-patient input** via form.
  * Perform **batch predictions** via CSV upload.
  * Show **probability estimates** of CKD.
  * Display **feature importance visualization**.

---

## 🎯 Achievements

* Implemented a **full ML pipeline**: data preprocessing → balancing → training → evaluation → deployment.
* Compared multiple classifiers and selected **XGBoost** as the best model.
* Built a **Streamlit app** for interactive predictions (single and batch).
* Learned hands-on ML deployment techniques.

---

## ⚠️ Disclaimer

This project is **for learning purposes only**. Predictions should **not** be used for medical or clinical decision-making.

---