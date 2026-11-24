# Diabetes Prediction System

A Machine Learning project that predicts Glycosylated Hemoglobin (GlyHb) levels and diabetes classification using patient health data.

## About the Project

This project applies machine learning techniques to patient health data to perform two prediction tasks:

- **Regression:** Predicts GlyHb percentage.
- **Classification:** Predicts the diabetes class (0 or 1).

Several machine learning algorithms are trained and compared for each task to evaluate their predictive performance.

## Main Features

- Data cleaning and missing-value handling.
- Feature engineering for regression (BMI and Waist-to-Hip Ratio).
- Numerical and categorical data preprocessing.
- Training and comparison of multiple machine learning algorithms.
- Model evaluation using appropriate performance metrics.
- Feature importance analysis.
- Saving and loading trained models.
- Interactive Gradio web application.

## Machine Learning Workflow

1. Data loading
2. Data cleaning
3. Feature engineering
4. Training and testing data splitting
5. Data preprocessing
6. Model training
7. Model evaluation and comparison
8. Feature importance and interpretation
9. Model saving
10. Gradio application

## Model Comparison Results

Multiple algorithms were compared for both regression and classification.

Among the regression models evaluated, **Random Forest achieved the best overall performance**, with the lowest MAE, MSE, and RMSE and the highest R² score.

For classification, **Random Forest achieved the highest F1-score** among the evaluated models.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost and CatBoost
- Matplotlib
- Gradio
- Joblib

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Leta-Kasahun/diabate-prediction.git
cd diabate-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

**Windows (Git Bash):**

```bash
source .venv/Scripts/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Place the dataset in:

```text
data/raw/diabetes.csv
```

The saved models and preprocessing pipelines are stored in:

```text
models/
├── regression/
│   └── best_regression_model.pkl
├── classification/
│   └── best_classification_model.pkl
└── preprocessing/
    ├── regression_preprocessor.pkl
    └── classification_preprocessor.pkl
```

## Run the Application

From the project root directory:

```bash
python app/gradio_app.py
```

Open the local URL displayed in your terminal.

## Author

Leta Kasahun  
Software Engineering Student

## Repository

[GitHub – Diabetes Prediction System](https://github.com/Leta-Kasahun/diabate-prediction)

## Note

This project is intended for educational and machine learning research purposes. Its predictions are not a substitute for professional medical diagnosis.