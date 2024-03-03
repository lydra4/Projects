#Imports
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Read Data Files
df = pd.read_csv('E:/GA_DSIF5 Copy/Personal/Projects/Student Score Prediction/Data/df_cleaned.csv', index_col='index')

#Train Test Split
X = df.drop(columns='final_test')
y = df['final_test']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  

#Normalization
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

#Linear Regression
lr = LinearRegression(n_jobs=-1)
lr.fit(X_train, y_train)
joblib.dump(value=lr, filename='E:/GA_DSIF5 Copy/Personal/Projects/Student Score Prediction/Data/linear_regression.pkl')

#Random Forest
rfr = RandomForestRegressor(n_jobs=-1, random_state=42)

param_grid = {'max_depth':[50,100,150,200], 
              'max_features':[1,2,3,4,5,6], 
              'min_samples_split':[30,40,50], 
              'min_samples_leaf':[30,40,50]
             }


gs_ss_rfr = GridSearchCV(estimator=rfr, 
                          param_grid=param_grid, 
                          cv=5, 
                          scoring='neg_mean_squared_error',
                          n_jobs=-1,
                          verbose=10) 

gs_ss_rfr.fit(X_train,y_train)

joblib.dump(value=gs_ss_rfr, filename='E:/GA_DSIF5 Copy/Personal/Projects/Student Score Prediction/Data/random_forest.pkl')

#Artificial neural network
model = Sequential()
model.add(layer=Dense(units=7, activation='relu'))
model.add(layer=Dense(units=20, activation='relu'))
model.add(layer=Dense(units=50, activation='relu'))
model.add(layer=Dense(units=20, activation='relu'))
model.add(layer=Dense(units=1))


model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x=X_train, 
          y=y_train, 
          epochs=20, 
          validation_data=(X_test, y_test))

model.save(filepath='E:/GA_DSIF5 Copy/Personal/Projects/Student Score Prediction/Data/ANN.keras')