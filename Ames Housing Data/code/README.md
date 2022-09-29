# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2 - Ames Housing Data and Kaggle Challenge

### Overview

This is our 2nd Project; we are analysing the Ames Housing Data Set and creating a regression model to predict the Sale Price of homes sold based on the given features. A little background, Ames is a city in the State of Iowa, USA that has a population of 67,000. Around half are students at the famed Iowa State University. With Ames being popular for its education, the rental property is a booming market.

We are provided with 2 datasets detailing the sale of home between the years 2006 to 2010. 1 titled as ‘Train.csv’ and the other ‘Test.csv’. The difference between the former and latter is the former contains the Sale Price while the latter lacks it. 

We are tasked with creating a regression model utilising the data in ‘Train.csv’ that generalises below 5%. Using the regression model, we predict the Sale Price of each home in ‘Test.csv’ and upload our predictions to Kaggle and obtain a Public and Private Score.

### Problem Statement

We will be taking on the role of real estate consultants providing recommendations for asset appreciation. To do that we have a 2-pronged approach, firstly identifying features that have a strong correlation with the sale price. Secondly, recognise neighbourhoods that can fetch the highest price.

---

### Datasets

#### Provided Data

Listed below are the datasets included in the [`data`](../data/) folder for this project. 

* [`train.csv`](../data/train.csv): Training data  
* [`test.csv`](../data/test.csv): Test Data 

#### Additional Data
* https://www.kaggle.com/c/dsi-us-11-project-2-regression-challenge 
* https://rdrr.io/cran/AmesHousing/man/ames_raw.html 

---

### Data Cleaning
In terms of Data Cleaning, Features that contain more than 50% missing values were dropped. For the remaining features that contain Null values they were either filled with a 0 if numerical or ‘None’ if it is a categorical Feature.

However, there is an exception to the rule. For the feature 'Lot Frontage', I make the assumption that Lot Frontage Area is Correlated with the respective neighbourhood and filled in Nan with the mean Lot Frontage area by neighbourhoods. 

### Analysis
We first came to the conclusion that some features are immutable. Non-Exhaustive Examples would be ‘Year Built’ and ‘Gr Liv Area’. However other features are variable and we analysed them. To do that we looked at the variable features that have the greatest positive correlation with Sale Price.

The rationale for that will be investors would know which feature to alter to increase the Sale Price. 

We also did a deep dive into the top 3 populated neighbourhoods and imposed our model. The rationale analysis on the top 3 neighbourhoods would be there is sufficient sample. 

---

### Conclusion and Recommendation

For property in Ames, Iowa, an investor can look into Total basement square feet, year remodified and garage area to maximise one’s investment.

Although there are other features that have a higher coefficient than the ones mentioned above, we have assumed that the features are fixed and unalterable.

However, to would be investors looking around the market, the flaw in the model is that there is no presence of negative coefficient(s). This does not affect reality and is a word of caution to potential customers.

Performing a deep dive into the neighbourhoods of CollgCr, OldTown and Somerst combined, the foundation, kitchen and pool area of homes in these areas would generate the greatest rise in Sale Price. The model used in this data set is Polynomial regression of degree 2, therefore investors looking to improve more than a feature can upgrade both the utilities and garage cars to maximise return on investment.

However a word of caution to investors, on the contrary improving the external quality of the house would lead to a depreciation of the asset.

