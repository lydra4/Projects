# AIAP Batch 16 Technical Assessment: Lung Cancer Occurrence

### Personal Information
Full name (as in NRIC): Ong Song Yi
Email: lydra4@hotmai.com

### Folder and structure
The folder contains 3 sub-folders titled, .github, data and src, within the folder itself there are 3 files, EDA.ipynb, README.md and requirements.txt

The first subfolder came default in the link, https://techassessment.blob.core.windows.net/aiap16-assessment-data/aiap16-NAME-NRIC.zip, which contains the repository template.
In the second sub-folder named data, there are 5 files detailed below:
- lung_cancer.db -> Initial Dataset
- df_cleaned.csv -> CSV file after Dataset has been cleaned and feature engineering is complete
- logistic_regression.pkl, random_forest_classifier.pkl, xgbclassifier.pkl ->p Machine learning models saved in pickle format

### Pipeline
The Pipeline is straight forward. Below are the steps of the pipleline in sequential order:

1) Data is read from the database file. Since there is only 1 table in the database file, only 1 table is read.
2) Next the data is cleaned. Missing values are first dealth with, next feature engineering is done and finally feature selection takes place.
3) Lastly, 3 models, Logistic Regression, XGBoost, Random Forest, are fitted on the dataset
4) Gridsearch is performed to find the optimal hyperparameters
5) Last but not least, the model is saved as a pickle file

### Overview

I will be taking on the role of Ministry Of Health (MOH), Singapore, to build Machine Learning Models for detection of Lung Cancer.

### Problem Statement

Globally, lung cancer is the leading cause of death for both genders claiming millions of lives yearly. Early detection is essential to boost survival rates. However current detection methods, Chest x-ray, CT Scan and MRI, can be costly or time consuming. Therefore to reduce cost and save time, I am tasked by MOH to build models that detect lung cancer.

### Datasets

The dataset is in the link below. The file in the link is a database with the extension, .db. In addition to that, there is only one table in the database.

https://techassessment.blob.core.windows.net/aiap16-assessment-data/lung_cancer.db

### Columns Cleaned
 ______________________________________________________________________
|        Feature         |     Issue     |         Method              |
| -----------------------|:-------------:| ---------------------------:|
| COPD History           |   1,112 NaN   | Impute with Mode from train |
| Taken Bronchodilators  |   1,061 NaN   | Impute with Mode from train |
| Air Pollution Exposure |     3 NaN     |        Drop rows            |
| Age                    |Negative Values|     Absolute column         |
| Start Smoking          |Not Applicable |  Change to value, 0         |
| Stop Smoking           |Not Applicable |  Change to value, 0         |
| Stop Smoking           |Still Smoking  |  Change to value, 2024      |
| Gender                 |    1 NaN      |        Drop Row             |
| Gender                 | Cap Strings   |        Capitalisation       |
| Dominant Hand          | RightBoth     |  Change to string, Both     |
 ----------------------------------------------------------------------

---

### Feature Engineering

Once the columns have been cleaned, several new features were engineered.
In addition, label encoder and one hot encoding were performed

1) Change in Weight = Current Weight - Last Weight
2) Years Smoke = Stop Smoking - Start Smoking
3) Air Pollution Exposure = Label Encoded
4) Frequency of Tiredness = Label Encoded
5) Gender = One Hot Encoding
6) COPD History = One Hot Encoding
7) Genetics Markers = One Hot Encoding
8) Taken Bronchodilators = One Hot Encoding
9) Dominant Hand = One Hot Encoding

---
### EDA
For feature selection, I separated the features into Numerical features and Categorical Features.

Numerical feature selection was performed using the ANNOVA F-Value while Categorical feature selection was done using Chi Squared test.

---
#### Hyparameter tuning

For the models, I have deicided to use Logistic Regression, Random Forest and XGBoost.

The reason for Logistic Regression is because it is the base model from which other models are compared against for a classification problem.

While for Random Forest, this was a widely used model during my data science boot camp.

And last but not least, XGBoost, the model which I use most frequently on a day to day setting.

Initially I wanted to optimize the recall metics while performing grid search. Recall is the ratio of correctly predicted positive cases against the number of actual positive cases.
This means that a false negative is more important than a false positive. Lung Cancer occurence not detected by the model is far more important than lung cancer occurence which are not in actual.

However after getting the ratio of positive against negative, 54.4% vs 45.6%, the data is marginally imbalanced. 
I deicided to turn class weights on for all 3 models.

---

### Conclusion and Recommendation
Since logistic regression is the base model against which other models are compared against, it lags behind the other 2 models, Random Forest and XGBoost, in all metrics.
Therefore my choice of model is Random Forest. Although it does marginally worst on the train set relative to XGBoost, the accuracy score of Random Forest for both train and test is closer. 

---