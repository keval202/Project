# 🎓 Student Exam Score Prediction using Machine Learning

## 📌 Project Overview
This project predicts student exam scores based on study habits, lifestyle, and personal factors using Machine Learning.

It includes:
- Data preprocessing using Pipeline & ColumnTransformer
- Model comparison (Linear, Random Forest, XGBoost, Decision Tree)
- Hyperparameter tuning using GridSearchCV
- Deployment using Streamlit

---

## 🚀 Live Features
- Predict student exam score instantly
- Clean and interactive UI (Streamlit)
- Real-time inference using trained ML pipeline

---

## 🧠 Machine Learning Workflow

1. Data Cleaning & Preprocessing  
2. Feature Engineering  
3. Train-Test Split  
4. Model Training  
5. Model Comparison  
6. Hyperparameter Tuning  
7. Final Model Selection (Linear Regression)  
8. Deployment with Streamlit  

---

## 📊 Model Performance

| Model              | R² Score | MAE   |
|--------------------|--------|------|
| Linear Regression  | **0.8927** | **4.28** |
| XGBoost            | 0.8749 | 4.70 |
| Random Forest      | 0.8481 | 5.02 |
| Decision Tree      | 0.7051 | 6.89 |

---

## ⚙️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  

---

## 📁 Project Structure

```bash
PredictingStudentExamScores/
│
├── data/
│   └── student_habits_performance.csv
├── model.pkl
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash

### 1. Create Virtual Environment
python -m venv venv

### 2. Activate Virtual Environment

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Train Model
python train_model.py

### 5. Run Streamlit App
python -m streamlit run app.py

```
---

## 🧠 Key Learnings

- Avoiding data leakage using Pipeline  
- Proper encoding strategies  
- Model selection using metrics  
- Deploying ML apps with Streamlit  

---

## ⭐ If you like this project
Give it a ⭐ on GitHub!
