# SignalGraph Churn Intelligence Platform

AI-powered customer churn prediction platform with Explainable AI (SHAP), risk segmentation, retention recommendations, and interactive Streamlit dashboards.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-MachineLearning-orange)
![SHAP](https://img.shields.io/badge/ExplainableAI-SHAP-green)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![PowerBI](https://img.shields.io/badge/PowerBI-Dashboard-yellow)

## Application Screenshots

### Customer Input Dashboard

![Dashboard](images/dashboard.png)

### Churn Prediction & Risk Analysis

![Prediction](images/prediction.png)

### Explainable AI (SHAP) Analysis

![SHAP](images/shap.png)

### Model Feature Importance

![Feature Importance](images/feature_importance.png)

## Project Overview

SignalGraph Churn Intelligence Platform is an end-to-end machine learning solution designed to identify customers at risk of churn and recommend proactive retention strategies.

The platform combines predictive analytics, Explainable AI (SHAP), and interactive visualizations to help businesses understand churn drivers, prioritize high-risk customers, and improve customer retention outcomes.

Key capabilities include:

- Customer churn prediction
- Risk categorization
- SHAP-based explainability
- Feature importance analysis
- Retention recommendation engine
- Interactive Streamlit dashboard

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/SignalGraph-Churn-Intelligence-Platform.git
```
Install dependencies:

```bash
pip install -r requirements.txt
```
Run the application:

```bash

streamlit run app/app.py
```

---

```markdown
## Repository Structure

```text
Customer-Churn-Project/
│
├── app/
│   └── app.py
│
├── dashboard/
│   └── PowerBI.pbix
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── images/
│   ├── dashboard.png
│   ├── prediction.png
│   ├── shap.png
│   └── feature_importance.png
│
├── model/
│   ├── columns.pkl
│   ├── model.pkl
│   └── scaler.pkl
│
├── notebook/
│   └── churn_analysis.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies Used

### Programming

* Python
* SQL

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Explainable AI

* SHAP

### Visualization

* Matplotlib
* Seaborn
* Plotly
* Power BI

### Deployment

* Streamlit

### Version Control

* Git
* GitHub

---

Open the terminal and run:

```bash

git status
