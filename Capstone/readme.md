# Capstone Project: Credit Card Fraud Detection

### Overview

This is my last project, I will be taking on the role of the data science team in the largest bank in Singapore, DBS.

### Problem Statement

The COVID19 Pandemic has accelerated e-commerce usage, this is has led to a rise in Credit Card Transactions. An increase in Credit Card transactions would cause more fraud transactions being committed. Therefore, the data science team would like to come up with an improved machine learning algorithm to detect fraud.

### Datasets

My dataset is in the link below. There are 2 datasets in the hyperlink, I will be using the dataset smaller in size due to physical limitations.

https://www.kaggle.com/datasets/kartik2112/fraud-detection 

---

### EDA

There will not be any data cleaning for this dataset as the information contained within is perfect thus, I will jump to EDA. The data is hugely imbalanced. For every 1,000 credit card transactions less than 4 are fraud, I will consider utilising SMOTE and turning on class weights in the machine learning algorithms. 

The main task of EDA will be to determine if the feature should be kept or dropped. If the feature is kept, the next step would be to determine if it should remain as numerical or categorical. For categorical columns, I will refer to the internet for the definition of how to segregate the feature.

---

### Feature Engineering

For my analysis, I utilise Random Forest and Logistic Regression as the machine learning models for the classification between the 2 subreddits.

However, in terms of feature engineering, I utilised both TFIDF vectorizer and Count Vectorizer. My rationale was that since I am only utilising 2 models, I should make up for it by increasing the ways I engage in feature engineering.

Once model training, hyperparameter tuning and model evaluation was done.
I noticed that logistic regression does better on accuracy relative to Random Forest. However, while it does well in accuracy there is a trade-off. Its generalization score is above the 5% acceptance range.

In this project, I had to compromise my generalization score to achieve better accuracy.

Having said that, for classification problem, accuracy is not the only metrics to measure the effectiveness of a machine learning algorithm. Precision and recall are the other measures of a model. Since, I have a balanced dataset, I will use accuracy as the main metric, precision and recall as the secondary measures.

---

#### Hyparameter tuning

In conclusion, logistic regression model on TFIDF Vectorized data gives the highest accuracy however its�s generalization score can be improved upon. These can be done but not limited to increasing sample size, utilising other classification machine learning models, using ensemble methods (Stacking, Bagging (I did bagging but it took 1 hour and not completed)) and raising the K value during cross validation.
