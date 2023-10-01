import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("insurance.csv",sep=",")
print(df)

df.head(5)

df.tail(5)

df.describe()

df.dtypes


le = LabelEncoder()

df.loc[:,['sex','smoker','region']] = \
df.loc[:,['sex','smoker','region']].apply(le.fit_transform)


x = df.drop(['expenses'],axis=1)
y = df['expenses']

df.corr()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
lr_score = r2_score(y_test,y_pred)


print(lr_score)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)
rf_score = r2_score(rf_pred,y_test)

print(rf_score)

import pickle
pickle.dump(rf,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[20,0,28.4,0,1,2]]))