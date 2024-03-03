# This module contains functions to read and clean the data
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import copy
import sys
import os

# Read data from db file taking 2 arguments, database directory and sql query
def read_data() -> pd.DataFrame:

    slice_string = sys.argv[0][:sys.argv[0].rfind('/')]
    another_sliced_string = slice_string[:slice_string.rfind('/')] + '/data/'
    file_name = [file for file in os.listdir(path=another_sliced_string) if '.db' in file][0]
    database = another_sliced_string + file_name

    con = sqlite3.connect(database=database)
    df = pd.read_sql_query(sql='select * from lung_cancer', con=con)

    return df

# Clean Data
def clean_data(df:pd.DataFrame) -> pd.DataFrame:

    # Drop Null rows in the column 'Air Pollution Exposure'
    df = df.dropna(axis=0, subset=['Air Pollution Exposure'])

    # Drop ID Column
    df = df.drop(columns='ID')

    # Convert negative age to positive
    df['Age'] = abs(df['Age'])

    # Feature engineer Change in Weight
    # Drop columns Current Weight and Last Weight
    df['Change in Weight'] = df['Current Weight'] - df['Last Weight']
    df = df.drop(columns=['Current Weight', 'Last Weight'])

    # Change 'Not Applicable' values in the columns, Stop Smoking and Start Smoking to the value 0
    df['Stop Smoking'] = df['Stop Smoking'].replace(to_replace='Not Applicable', value=0)
    df['Start Smoking'] = df['Start Smoking'].replace(to_replace='Not Applicable', value=0)

    # Replace 'Still Smoking' values in the column, Still Smoking
    df['Stop Smoking'] = df['Stop Smoking'].replace(to_replace='Still Smoking', value=2024)

    # Convert columns, Still Smoking and Start Smoking, to integer
    df['Start Smoking'] = df['Start Smoking'].astype(int)
    df['Stop Smoking'] = df['Stop Smoking'].astype(int)

    # Feature engineer Years Smoke
    # Drop columns Stop Smoking and Start Smoking
    df['Years Smoke'] = df['Stop Smoking'] - df['Start Smoking']
    df = df.drop(columns=['Start Smoking', 'Stop Smoking'])

    # Remove values, 'NAN', in gender column
    df = df[df['Gender'] != 'NAN']

    # Capitalisation all caps values in gender column
    df['Gender'] = df['Gender'].str.replace(pat='FEMALE', repl='Female', case=True)
    df['Gender'] = df['Gender'].str.replace(pat='MALE', repl='Male', case=True)

    # Changing value, RightBoth, in Dominant Hand column to Both
    df['Dominant Hand'] = df['Dominant Hand'].str.replace(pat='RightBoth', repl='Both')

    # Train test slit, using test_size and random_state values from EDA, before imputing mode in columns, Taken Bronchodilators and COPD History
    X = df.drop(columns='Lung Cancer Occurrence')
    y = df['Lung Cancer Occurrence']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    COPD_History_mode = X_train['COPD History'].mode(dropna=True)[0]
    Taken_Bronchodilators_mode = X_train['Taken Bronchodilators'].mode(dropna=True)[0]

    df['COPD History'] = df['COPD History'].fillna(value=COPD_History_mode)
    df['Taken Bronchodilators'] = df['Taken Bronchodilators'].fillna(value=Taken_Bronchodilators_mode)

    # Feature Selection
    # Removing columns
    df = df.drop(columns=['Age', 'Years Smoke', 'Taken Bronchodilators', 'COPD History', 'Dominant Hand', 'Frequency of Tiredness'])
    
    # Label Encode the column, Air Pollution Exposure
    df['Air Pollution Exposure'] = df['Air Pollution Exposure'].replace(to_replace='Low', value=1)
    df['Air Pollution Exposure'] = df['Air Pollution Exposure'].replace(to_replace='Medium', value=2)
    df['Air Pollution Exposure'] = df['Air Pollution Exposure'].replace(to_replace='High', value=3)

    # One Hot encode remaining categorical features
    df = pd.get_dummies(data=df, drop_first=True)

    # Performing train test split before normalizing column, Change in Weight, using Min Max Scaler
    X = df.drop(columns='Lung Cancer Occurrence')
    y = df['Lung Cancer Occurrence']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    temp_df = pd.DataFrame(copy.deepcopy(X_train['Change in Weight']))
    mms = MinMaxScaler()
    mms.fit(temp_df)
    df['Change in Weight'] = mms.transform(df[['Change in Weight']])

    return df

def read_and_clean_data() -> pd.DataFrame:
    df = read_data()
    df = clean_data(df=df)

    return df