# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Web APIs & NLP

### Overview

In our 3rd project, we are tasked with scrapping 2 subreddits and training 2 models (1 model must be Random Forest) to differentiate materials between the 2 subreddits.

### Problem Statement

We are tasked to scrap 2 subreddits and train 2 models in order to differentiate materials from the 2 subreddits.

---

### Datasets

Below are the hyperlinks to the data gathered from the 2 subreddits:

Python: https://www.reddit.com/r/Python/
JavaScript: https://www.reddit.com/r/javascript/ 

---

### Data Cleaning

Data cleaning for this project can be categorised into 4 broad categories. Missing Values (NaN), Emoticons, Removed posts by moderators and foreign languages.

For the first 3, I have found and replaced them with a blank. Regarding the last part of data cleaning, foreign language, firstly I had to detect if the language is in English. Secondly, if the language is not English, translation to English had to be done.

In all my data cleaning methods, I preserve to keep the data as much as possible.

---

### Analysis

For my analysis, I utilise Random Forest and Logistic Regression as the machine learning models for the classification between the 2 subreddits.

However, in terms of feature engineering, I utilised both TFIDF vectorizer and Count Vectorizer. My rationale was that since I am only utilising 2 models, I should make up for it by increasing the ways I engage in feature engineering.

Once model training, hyperparameter tuning and model evaluation was done.
I noticed that logistic regression does better on accuracy relative to Random Forest. However, while it does well in accuracy there is a trade-off. Its generalization score is above the 5% acceptance range.

In this project, I had to compromise my generalization score to achieve better accuracy.

Having said that, for classification problem, accuracy is not the only metrics to measure the effectiveness of a machine learning algorithm. Precision and recall are the other measures of a model. Since, I have a balanced dataset, I will use accuracy as the main metric, precision and recall as the secondary measures.

---

### Conclusion and Recommendation

In conclusion, logistic regression model on TFIDF Vectorized data gives the highest accuracy however its’s generalization score can be improved upon. These can be done but not limited to increasing sample size, utilising other classification machine learning models, using ensemble methods (Stacking, Bagging (I did bagging but it took 1 hour and not completed)) and raising the K value during cross validation.
