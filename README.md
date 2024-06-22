# Overview of the Analysis

## Purpose of Analysis
The purpose of this analysis is to build and evaluate a predictive model to identify the creditworthiness of borrowers. By leveraging historical lending data from a peer-to-peer lending services company, the aim is to predict whether a given loan will be healthy (low risk) or at high risk of default. This analysis will use supervised learning to assist in making informed lending decisions, ultimately reducing financial risk for the lending company.

## Financial Information Use for Predictions
The dataset contains financial information related to various loans, including:

- loan_size: The amount of the loan.
- interest_rate: The interest rate applied to the loan.
- borrower_income: The income of the borrower.
- debt_to_income: The ratio of the borrower's total debt to their income.
- num_of_accounts: The number of accounts held by the borrower.
- derogatory_marks: The number of derogatory marks on the borrower's credit report.
- total_debt: The total debt of the borrower.
- loan_status: The status of the loan (0 for healthy, 1 for high risk).

The primary objective is to predict the loan status of each loan based on the financial variables listed above other than 'loan_status' (i.e., 'features').

## Predicting Loan Status
The target variable 'loan_status' indicates the health of the loan, with 0 indicating a healthy loan (low risk) and 1 indicating a high-risk loan.

I used value_counts() to understand the distribution of loan statuses across the dataset. This indicated that out of 77,536 loans in the data set, 75,036 (96.8%) were healthy loans and 2,500 (3.2%) were high-risk loans.


## Stages of Machine Learning

### Data Preparation
1) Read lending_data.csv into a Pandas Data Frame.
2) Separated target variable 'loan_status' from the rest of the financial variables in the data frame, creating labels and features respectively.
3) Used 'train_test_split' from sklearn to divide the dataset into training and testing sets.

### Model Creation and Training
Used logistic regression model (i.e., LogisticRegressionform sklearn) for binary classification task. Used model.fit to train the model with training data (i.e., X_train and y_train).

### Model Evaluation
Used model.predict to predict using X_test. The results were then evaluated using hte confusion matrix and classification report. The confusion matrix provides a matrix of the true positives, true negatives, false positives, and false negatives while the classificaiton report includes the precision, recall, f-1, and accuracy scores for each loan status class (healthy or high-risk).


# Results

## Confusion Matrix

- The confusion matrix reveals a small number of misclassifications, with more false positives (102) than false negatives (56). This suggests that the model errs on the side of caution, favoring the identification of high-risk loans.

## Classification Report

- The model demonstrates a very high accuracy of 99%, indicating strong effectiveness in classifying loans correctly.
- For healthy loans, the precision, recall, and F1-score are nearly perfect, showcasing the model's exceptional capability in identifying healthy loans accurately.
- For high-risk loans, the precision is slightly lower at 85%, but the recall is high at 91%. This indicates that the model is proficient at identifying most high-risk loans, though it does result in some false positives.


# Summary

I recommend this linear regression model due to its impressive overall accuracy of 99% in classifying loans correctly. Our goal is to determine the creditworthiness of borrowers, making a high precision score crucial in predicting high-risk loans. This is because labeling a loan as high-risk can have significant consequences, such as loss of business from loan rejections. Additionally, a high recall score is important if the cost of loan defaults outweighs the cost of lost business. While this model already excels in accuracy and F1-scores, it can be further fine-tuned to increase the precision score from 85% in predicting high-risk loans.

