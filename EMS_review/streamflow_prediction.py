import pandas as pd 
import os 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

# Set the path to the directory containing the data files
path = fr"D:\MyDataBase\Streamflow_data_stations\STREAMFLOW data"

# Get the list of file names in the directory
NAMES = os.listdir(path)

# Select the first 10 stations that end with ".csv"
NAMES = [name for name in NAMES if name.endswith(".csv")][:10]
print(NAMES)

# Read the data from the first file in the list
data = pd.read_csv(os.path.join(path, NAMES[0]))   # columns: date, streamflow
print(data.head())  

# Remove rows with negative streamflow values
data[data.streamflow < 0] = np.nan
data = data.dropna()
print(data.head())

# Convert the data to monthly frequency
data.index = pd.to_datetime(data['date'])
data = data.resample('M').agg({'streamflow':'mean'})

# Add month and year columns
data['month'] = data.index.month
data['year'] = data.index.year
data[data.year>2002] = np.nan
data[data.year<1979] = np.nan
data = data.dropna()
# drom date column
print(data.head())

# Split the data into features (X) and target (y)
X = data.drop('streamflow', axis=1)
y = data['streamflow']

# Define the range for training and testing data
train_range = ((data['year'] < 1990) & (data['year'] > 1980)) # Change the year as per your requirement
test_range = ((data['year'] < 2002) & (data['year'] > 1990))    # Change the year as per your requirement

# Split the data into training and testing sets based on the range
X_train = X[train_range]
y_train = y[train_range]
X_test = X[test_range]
y_test = y[test_range]

print(X_train.head())

# Instantiate the random forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model using root mean squared error (RMSE) and Nash-Sutcliffe Efficiency (NSE)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
nse = 1 - sum((y_test - y_pred)**2) / sum((y_test - y_test.mean())**2)
print(nse)
print(rmse)

# Plot the observed vs predicted values
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.subplot(1, 2, 1)
# set figure siz
plt.scatter(y_test, y_pred)
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.text(0.05, 0.90, f"NSE = {nse:.2f}\nRMSE = {rmse:2.4}", transform=plt.gca().transAxes, fontsize=14, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
plt.grid()

# Plot the time series of predicted and observed values
plt.subplot(1, 2, 2)
y_pred_all = model.predict(X)
plt.plot(data.index, y, label='Observed')
plt.plot(data.index[train_range], y_pred_all[train_range], label='Predicted (Training)', color='orange')
plt.plot(data.index[test_range], y_pred_all[test_range], label='Predicted (Testing)', color='red')
plt.title("Random Forest monthly streamflow prediction")
plt.xlabel("Date")
plt.ylabel("average monthly streamflow (cfs)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("/data/MyDataBase/SWATGenXAppData/codes/EMS_review/Streamflow_prediction.jpeg", dpi=300)
plt.show()
