# Module for Data Analysis & Visualization
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint

# Module for Machine Learning
from sklearn.linear_model import LinearRegression  # Linear Regression
from sklearn.model_selection import train_test_split  # Splitting data into training set and test set
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures

# For Download Data From Taiwan CDC
import requests
import urllib3


# Importing the dataset
dataset = pd.read_csv('Taiwan_Covid19_Cases.csv')
# Feature Engineering: Convert 'Date' to pandas datetime and calculate 'Day' as the number of days since the first date
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Day'] = (dataset['Date'] - dataset['Date'].min()).dt.days
# Feature selections
idx = 500
x = dataset['Day'].iloc[idx:]
y = dataset['Total_Cases'].iloc[idx:]

x = np.array(x).reshape(-1, 1)  # Convert to a 2D array
y = np.array(y)

data_len = len(x)

split_rate = 0.8
# Splitting data into training set and test set
x_train = x[0: int(data_len * split_rate)]
y_train = y[0: int(data_len * split_rate)]

x_test = x[int(data_len * split_rate):]
y_test = y[int(data_len * split_rate):]


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

# Linear Regression
x_grid = np.arange(min(x), max(x)+100, 0.1)  # to make each day in 10 pieces(11410)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.figure(figsize=(12, 6))
plt.scatter(x, y, color='red', label="Actual Cases")
# plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color='blue', label="Polynomial Prediction")
plt.plot(x, lin_reg.predict(x), color='green', label="Linear Regression")
plt.grid(True)
plt.title('COVID-19 Daily Cases and Predicted Future Cases in Taiwan(Linear Regression) - Training Set')
plt.xlabel('Date')
plt.ylabel('Daily COVID-19 Cases')
plt.legend()
plt.show()


# Testing set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_test, y_test)
print(lin_reg.score(x_test, y_test))

x_grid = np.arange(min(x), max(x)+100, 0.1)  # to make each day in 10 pieces(11410)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.figure(figsize=(12, 6))
plt.scatter(np.arange(0, len(x_test)), y_test, color='red', label="Actual Cases")
# plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color='blue', label="Polynomial Prediction")
plt.plot(np.arange(0, len(x_test)), lin_reg.predict(x_test), color='green', label="Linear Regression")
plt.grid(True)
plt.title('COVID-19 Daily Cases and Predicted Future Cases in Taiwan(Linear Regression - Test Set')
plt.xlabel('Date')
plt.ylabel('Daily COVID-19 Cases')
plt.legend()
plt.show()