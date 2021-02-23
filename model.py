import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# read in data as pandas dataframe
data = pd.read_csv('adult.data')
data.columns = ['age','workclass','fnlwgt','education','education-num','marital-status',
'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']

# one hot encode categorical data
# simplify marital status into married, unmarried
marital_status_encoded = []
for status in data["marital-status"]:
    if status in ["Married-civ-spouse","Married-spouse-absent", "Married-AF-spouse"]:
        marital_status_encoded.append(1)
    else:
        marital_status_encoded.append(0)

# construct encoded data
copied_data = data[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','workclass','occupation','race','sex','salary']].copy()
encoded_data = pd.get_dummies(copied_data,columns=['workclass','occupation','race','sex','salary'], drop_first=True)
encoded_data["marital_status"] = pd.DataFrame(marital_status_encoded)

# split into train and test data
# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(encoded_data.drop('salary_ >50K', axis=1), encoded_data['salary_ >50K'], test_size=0.20, random_state=0)


# scale data because magnitude changes
standardscaler = StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_test = standardscaler.fit_transform(X_test)

# train
logreg = LogisticRegression().fit(X_train, y_train)

# evaluate accuracy of logistic regression
scores = logreg.score(X_test,y_test)
print(scores)
