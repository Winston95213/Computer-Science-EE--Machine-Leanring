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

# Fitting Linear Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)  # degree control the number of features
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)



# Polynomial Regression
x_grid = np.arange(min(x), max(x)+100, 0.1)  # to make each day in 10 pieces(11410)
x_grid = x_grid.reshape(len(x_grid), 1)
pred = lin_reg_2.predict(poly_reg.fit_transform(x_grid))
plt.figure(figsize=(12, 6))
plt.scatter(x, y, color='red', label="Actual Cases")
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color='blue', label="Polynomial Prediction")
plt.plot(x, lin_reg.predict(x), color='green', label="Linear Regression")
plt.grid(True)
plt.title('COVID-19 Daily Cases and Predicted Future Cases in Taiwan(Polynomial Regression)')
plt.xlabel('Date')
plt.ylabel('Daily COVID-19 Cases')
plt.legend()
plt.show()


train_result = []
test_result = []

plt.figure(figsize=(12, 6))
for i in range(2, 14):
    # Fitting Linear Regression to the dataset
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree=i)  # degree control the number of features

    x_train_poly = poly_reg.fit_transform(x_train)
    x_test_poly = poly_reg.fit_transform(x_test)

    # model learning
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(x_train_poly, y_train)

    # model predict
    train_predict = lin_reg_2.predict(x_train_poly)
    test_predict = lin_reg_2.predict(x_test_poly)

    # performance index calculation
    # Correlation Calculation
    # https://realpython.com/numpy-scipy-pandas-correlation-python/
    train_correlation = np.corrcoef(y_train, train_predict)[0,1]
    test_correlation = np.corrcoef(y_test, test_predict)[0,1]

    train_result.append(train_correlation)
    test_result.append(test_correlation)



    # train result
    # Polynomial Regression
    x_grid = np.arange(min(x_test), max(x_test)+100, 0.1)  # to make each day in 10 pieces(11410)
    x_grid = x_grid.reshape(len(x_grid), 1)
    #
    if i == 2:
        plt.plot(np.arange(0, len(x_test)), y_test, label="Actual Cases", color="r", linewidth=2)

    plt.plot(np.arange(0, len(x_test)), lin_reg_2.predict(x_test_poly), label="degree=" + str(i))
    # plt.plot(x_train, lin_reg.predict(x_train), color='green', label="Linear Regression")

    plt.grid(True)
    plt.title('COVID-19 Daily Cases and Predicted Future Cases in Taiwan(test set)')
    plt.xlabel('Date')
    plt.ylabel('Daily COVID-19 Cases')
    plt.legend()
    del lin_reg_2

plt.show()

train_result = []
test_result = []

plt.figure(figsize=(12, 6))
for i in range(2, 14):
    # Fitting Linear Regression to the dataset
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree=i)  # degree control the number of features

    x_train_poly = poly_reg.fit_transform(x_train)
    x_test_poly = poly_reg.fit_transform(x_test)

    # model learning
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(x_train_poly, y_train)

    # model predict
    train_predict = lin_reg_2.predict(x_train_poly)
    test_predict = lin_reg_2.predict(x_test_poly)

    # performance index calculation
    # Correlation Calculation
    # https://realpython.com/numpy-scipy-pandas-correlation-python/
    train_correlation = np.corrcoef(y_train, train_predict)[0,1]
    test_correlation = np.corrcoef(y_test, test_predict)[0,1]

    train_result.append(train_correlation)
    test_result.append(test_correlation)



    # train result
    # Polynomial Regression
    x_grid = np.arange(min(x_train), max(x_train)+100, 0.1)  # to make each day in 10 pieces(11410)
    x_grid = x_grid.reshape(len(x_grid), 1)
    #
    if i == 2:
        plt.plot(np.arange(0, len(x_train)), y_train, label="Actual Cases", color="r", linewidth=2)

    plt.plot(np.arange(0, len(x_train)), lin_reg_2.predict(x_train_poly), label="degree=" + str(i))
    # plt.plot(x_train, lin_reg.predict(x_train), color='green', label="Linear Regression")

    plt.grid(True)
    plt.title('COVID-19 Daily Cases and Predicted Future Cases in Taiwan(train set)')
    plt.xlabel('Date')
    plt.ylabel('Daily COVID-19 Cases')
    plt.legend()
    del lin_reg_2

plt.show()
