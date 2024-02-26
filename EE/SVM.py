# Module for Data Analysis & Visualization
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint

# Module for Machine Learning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# For Download Data From Taiwan CDC
import requests
import urllib3


def import_data():
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Suppress the warning
    url = 'https://od.cdc.gov.tw/eic/covid19/covid19_tw_specimen.csv'  # URL of Data From Taiwan CDC
    req = requests.get(url, verify=False)  # Get Data From URL
    url_content = req.content
    with open('Download_File.csv', 'wb') as f:  # Write Data to CSV File
        f.write(url_content)


def reformat_data():
    # Reformat Data
    with open('Download_File.csv', 'r') as f:
        reader = csv.DictReader(f)  # Read CSV File
        total_cases = 0  # Total Cases
        data = []
        count = 0

        for row in reader:  # Read Data in Row
            count += 1
            if count >= 0:
                daily_cases = int(row['Total'])  # Daily Cases
                date = row['通報日']  # Date
                total_cases += daily_cases  # Total Cases(Add Daily Cases)
                data.append({"Date": date, "Daily_Cases": daily_cases, "Total_Cases": total_cases})

    with open("Taiwan_Covid19_Cases.csv", "w", newline='') as f:
        fields_name = ["Date", "Daily_Cases", "Total_Cases"]  # Set Field Name
        writer = csv.DictWriter(f, fieldnames=fields_name)  # Write CSV File
        writer.writeheader()
        writer.writerows(data)


def feature_selection() -> (np.ndarray, np.ndarray, pd.DataFrame):
    # Importing the dataset
    dataset = pd.read_csv('Taiwan_Covid19_Cases.csv')
    # Feature Engineering: Convert 'Date' to pandas datetime and calculate 'Day' as the number of days since the first date
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset['Day'] = (dataset['Date'] - dataset['Date'].min()).dt.days
    # Feature selections
    idx = 500
    x = dataset['Day'].iloc[idx:]
    y = dataset['Total_Cases'].iloc[idx:]

    # Reshape x and y
    x = np.array(x).reshape(-1, 1)  # Convert to a 2D array
    y = np.array(y)

    return x, y, dataset


def split_data(x, y) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    # Splitting data into training set and test set
    data_len = len(x)

    split_rate = 0.8
    # Splitting data into training set and test set
    x_train = x[0: int(data_len * split_rate)]
    y_train = y[0: int(data_len * split_rate)]

    x_test = x[int(data_len * split_rate):]
    y_test = y[int(data_len * split_rate):]
    # set test_size as 0.2 is split 80% of the data to training set and 20% of the data to test set
    # random_state=0 is to fix the random split to the same every time
    return x_train, x_test, y_train, y_test


def predict_future_cases(model, dataset) -> (pd.DatetimeIndex, np.ndarray):
    # Predict future cases (e.g., 100 days into the future)
    future_days = pd.DataFrame({'Day': range(dataset['Day'].max() + 1, dataset['Day'].max() + 101)})
    # To transfer the future days to datetime
    future_dates = pd.date_range(start=dataset['Date'].max() + pd.Timedelta(days=1), periods=100, freq='D')
    future_days_array = future_days.values.reshape(-1, 1)  # Convert to a 2D array
    future_cases = model.predict(future_days)  # Predict future cases
    future_cases = future_cases.astype(int)  # Convert to int
    print("Predicted future cases:")
    print(future_cases)

    return future_dates, future_cases


def plot_data(dataset, future_dates, future_cases, y, model, x, x_train, x_test, y_train, y_test):
    # Plot the data,
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(500, len(x_train)+500), model.predict(y_train.reshape(-1, 1)), label='Predicted Daily Cases', color='b', marker=".")
    feature_array = [i for i in range(x.max() + 1, x.max() + 101)]
    # plt.scatter(feature_array, future_cases, label='Predicted Future Cases', color='g', marker=".")
    plt.plot(x_train, y_train, label='Actual Daily Cases', color='r', marker='.')
    plt.xlabel('Date')
    plt.ylabel('Daily COVID-19 Cases')
    plt.title('COVID-19 Daily Cases and Predicted Future Cases(SVM)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():

    # write a try and except if file not found then download data
    try:
        reformat_data()
    except FileNotFoundError:
        print("File Not Found")
        import_data()

    x, y, dataset = feature_selection()
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Create training model
    model = SVR(kernel='rbf', C=100, gamma="auto", epsilon=.1)
    model.fit(x_train, y_train)
    # Test training model
    x_pred = model.predict(x_train)

    # print("Predicted Values", y_pred)

    future_dates, future_cases = predict_future_cases(model, dataset)
    plot_data(dataset, future_dates, future_cases, y, model, x, x_train, x_test, y_train, y_test)

    # plot_data(dataset, future_dates, future_cases, y, model, x, x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()