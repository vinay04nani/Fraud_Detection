# Fraud_Detection

This repository contains a Jupyter Notebook implementing a machine learning model for fraud detection using Logistic Regression and Random Forest Classifier algorithms. The model is trained on a dataset of financial transactions to predict whether a transaction is fraudulent or legitimate.

# Dataset Requirements
The code expects a CSV file named Fraud.csv containing the following features:
# Transaction ID:
A unique identifier for each transaction.
# Amount: 
The financial amount of the transaction.
# Time: 
The timestamp of the transaction.
# Source: 
The source of the transaction (e.g., card, bank transfer).
# Destination: 
The destination of the transaction (e.g., merchant, account).
# Other relevant features: 
Additional features that may be relevant for fraud detection (e.g., location, customer information).
Make sure your dataset adheres to this format before running the code.

# Data Preprocessing:

# Cleans the data by:
Removing unnecessary columns like nameOrig, nameDest, and step.
Handling missing values (shown as nulls).
Applies Label Encoding to convert categorical features like type into numerical values.
Detects and removes outliers in numerical features to improve model training.
Model Training:

# Trains two models:
Logistic Regression: A statistical model for binary classification, using balanced class weights.
Random Forest Classifier: An ensemble model with 100 decision trees, also using balanced class weights.
Splits the data into training and testing sets.
Evaluation:

Evaluates both models using accuracy, precision, recall, F1-score, and confusion matrix.
Reports the performance metrics for each model.
Running the Code
Prerequisites:

Python 3.x
# Required libraries (listed in the notebook):
pandas
numpy
sklearn.model_selection
sklearn.preprocessing
sklearn.linear_model
sklearn.metrics

# Key Findings
The code explores two machine learning algorithms for fraud detection.
Both Logistic Regression and Random Forest achieve high accuracy in this example.
The notebook highlights the importance of handling imbalanced datasets by using balanced class weights.
It identifies features like amount, oldbalanceOrg, newbalanceOrg, and type as potentially relevant for fraud detection.

# Future Work
Experiment with hyperparameter tuning for both models.
Explore feature selection techniques to identify the most impactful features.
Consider using ensemble methods like XGBoost or LightGBM.
Investigate deep learning approaches for potentially complex fraud patterns.

# Disclaimer
This code is for educational purposes only and may not be suitable for production use. The performance of the model depends heavily on the quality of the data and may require further tuning for real-world applications.







