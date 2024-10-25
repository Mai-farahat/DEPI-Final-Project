# **🎓 Customer Churn Analysis & Prediction**

### Predicting and preventing customer churn through data-driven insights

## **📋 Project Overview**

### Customer churn is a critical issue for businesses, as losing customers can significantly impact revenue and growth. In this graduation project, we use advanced machine learning techniques to analyze customer churn and build a prediction model to help businesses identify customers at risk of leaving.

### By leveraging various data-driven strategies, we aim to provide actionable insights that help businesses reduce churn rates and improve customer retention.

## **🎯 Key Objectives**
- Understand the factors contributing to customer churn through exploratory data analysis.
- Build machine learning models that predict whether a customer will churn based on historical data.
- Deploy the best-performing model to provide real-time churn predictions.

## **🔍 Problem Statement**

### Businesses often face the challenge of customer attrition. Predicting which customers are likely to churn enables businesses to intervene before it’s too late. The goal is to develop a solution that can accurately predict churn and help decision-makers target at-risk customers with personalized strategies.

## **📊 Dataset Description**

### The dataset used in this project contains key information such as:

- Demographics: Gender, age, location
- Account Information: Contract type, tenure, payment method
- Service Usage: Products used, monthly charges
- Churn Label: Binary label indicating whether a customer churned or stayed


## **🛠 Project Workflow**

### This project follows a structured machine learning workflow, outlined as follows:

1. Data Collection: Gathering the necessary data for customer churn.
2.	Data Preprocessing: Handling missing values, outlier detection, feature scaling, and transformation.
3.	Exploratory Data Analysis (EDA): Visualizing key patterns and relationships between features.
4.	Model Development:
    - Logistic Regression
	- Decision Trees
	- Random Forest
	- XGBoost
5.	Model Evaluation: Assessing models using metrics like accuracy, precision, recall, F1-score, and AUC-ROC curve.
6.	Deployment: Selecting and deploying the most accurate model.

## **📈 Key Insights**

- High churn risk is observed among customers with shorter tenures and those using specific payment methods.
- Monthly charges and contract type have a significant impact on customer churn predictions.
- The XGBoost model provided the most accurate results in predicting customer churn, with an accuracy of 95%.

## **🛠 Technologies Used**

- Python: Programming language
- Pandas, NumPy: Data manipulation and analysis
- Matplotlib, Seaborn: Data visualization
- Scikit-learn: Machine learning models
- Jupyter Notebook: For organizing the workflow
- Flask & Streamlit : For Deployment