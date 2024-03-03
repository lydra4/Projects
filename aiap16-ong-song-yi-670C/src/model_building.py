from sklearn.base import BaseEstimator

import pandas as pd

from sklearn.model_selection import train_test_split

from read_and_clean_data import read_and_clean_data

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

import joblib

class model_building(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, x, y):
        return self
    
    def transform(self):
        
        df = read_and_clean_data()

        return df
    
    def machine_learning(self, df:pd.DataFrame):

        # Performing train test split using test size and random state values from EDA jupyter notebook
        X = df.drop(columns='Lung Cancer Occurrence')
        y = df['Lung Cancer Occurrence']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 1st Model: Logistics Regression
        logistics_regression = LogisticRegression()

        logistics_regression_parameters = {
                                            "class_weight": ["balanced"],
                                            "max_iter": [100000],
                                            "C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ],
                                            }
        
        # 2nd Model: XGBClassifier, I have to compute class weights as there is no class weight model's parameter
        class_weights = (y == 0).sum() / (y == 1).sum()
        
        xgboost_classifier = XGBClassifier(scale_pos_weight = class_weights, random_state=42)

        xgboost_classifier_parameters = {
                                         "eta": [0.1,0.2,0.3,0.4,0.5],
                                         "max_depth": [6,7,8,9,10],
                                         }
        
        # 3rd Model: Random Forest Classifier
        random_forest_classifier = RandomForestClassifier(random_state=42)

        random_forest_parameters = { 
                                    'max_depth':[50,100,150,200], 
                                    'min_samples_split':[30,40,50], 
                                    'class_weight':['balanced']
                                    }

        gridsearch_dictionary = {logistics_regression: logistics_regression_parameters,
                                 xgboost_classifier: xgboost_classifier_parameters,
                                 random_forest_classifier: random_forest_parameters}
        
        # Setting a list of names to save the model
        models_name = ['logistic_regression', 'xgbclassifier', 'random_forest_classifier']

        # Performing grid search over all the models and their parameters
        for key, item in gridsearch_dictionary.items():
            gridsearch = GridSearchCV(estimator=key,
                                      param_grid=item,
                                      scoring='accuracy',
                                      cv=5,
                                      n_jobs=-1,
                                      verbose=1)
            
            gridsearch.fit(X=X_train, y=y_train)

            position = list(gridsearch_dictionary).index(key)
            model_name = f'E:/GA_DSIF5 Copy/Personal/Projects/aiap16-ong-song-yi-670C/data/{models_name[position]}.pkl'
            joblib.dump(value=gridsearch, filename=model_name)

if __name__ == '__main__':

    model_building = model_building()
    df = model_building.transform()
    model_building.machine_learning(df=df)