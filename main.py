import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
df=pd.read_csv(r"D:\coding journey\aiml\python\udemy\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 6 - Polynomial Regression\Python\Position_Salaries.csv")
x=df.iloc[:, 1:-1].values
y=df.iloc[:, -1].values
regressor=RandomForestRegressor(n_estimators=50,random_state=0)
regressor.fit(x,y)
print(regressor.predict([[6.5]]))





