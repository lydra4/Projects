# Capstone Project: Credit Card Fraud Detection

### Overview

This is my last project at General Assembly, I will be taking on the role of the data science team at the largest bank in Singapore, DBS.

### Problem Statement

The COVID19 Pandemic has accelerated e-commerce usage, this has led to a rise in Credit Card Transactions. An increase in Credit Card transactions would lead to more fraud transactions being committed. Therefore, the data science team would like to come up with an improved machine learning algorithm to detect fraud.

### Datasets

My dataset is in the link below. There are 2 datasets in the hyperlink, I will be using the dataset smaller in size due to physical limitations.

https://www.kaggle.com/datasets/kartik2112/fraud-detection 

---

### EDA

There will not be any data cleaning for this dataset as the information contained within is perfect thus, I will jump to EDA. The data is hugely imbalanced. For every 1,000 credit card transactions less than 4 are fraud, I will consider utilising SMOTE and turning on class weights in the machine learning algorithms. 

The main task of EDA will be to determine if the feature should be kept or dropped. If the feature is kept, the next step would be to determine if it is numerical or categorical. For categorical columns, I will refer to the internet for the definition of how to segregate the feature.

---

### Feature Engineering

The main task of feature engineering is to determine if the column should be dropped or kept and the only factor in determining that is to observe if there is any relationship between that feature and the target label. Once the decision has been made, the task was to determine if a numerical feature is to remain numerical or be converted to categorical. This was done by analysing whether there is a correlation between the magnitude of the said feature and the target label. Magnitude of features with a zero or low correlation with the target label were converted to categorical columns.

With the above tasks completed, the next step is to investigate multicollinearity. There are multiple ways to investigate this issue and my statistical technique of choice is variance inflation factor (VIF). Thankfully there was low/no presence of multicollinearity with all the VIF scores below 5.

At the end all the features bar one was categorical, one feature was numerical. I chose to model on two data sets, one normalized with standard scaler and the other normalized with min max scaler.

---

#### Hyparameter tuning

While forming the idea of my capstone project, I have decided to utilise 4 models. They are logistic regression, random forest, LightGBM and last but not least, utilising the best hyperparameters from each model, stacking.

Utilising ROC, PR curves and more, model performance was evident. LightGBM did the best in recall scores on both train and test data. In addition, generalisation score was also fulfilled. This was followed closely by Random Forest model. Logistic Regression fared the worst against the above said models. Stacking was pulled down by the Logistic Regression model. Therefore, my model of choice is LightGBM performed on data normalized with Standard Scaler.

---

### Conclusion and Recommendation
LightGBM does the best among the other models in terms of recall score on test data. This is followed by Random Forest however this model does not satisfy the generalisation score of 5%. Logistic Regression and Stacking fared badly when compared to the 2 said models. The performance of Logistic Regression pulls down the scores of Stacking. LightGBM trained on data normalized with standard scaler is my model of choice. 