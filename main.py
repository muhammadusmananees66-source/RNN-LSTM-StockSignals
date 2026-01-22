from tensorflow import keras   # Keras used for building and training neural networks
from sklearn.preprocessing import StandardScaler   # standardizes numerical features by scaling them to have mean and variance
import numpy as np  # Numerical python to better handle vast amounts of numerical data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # advanced version of matplotlib with better tools for visualizations
import os 
from datetime import datetime

# used to suppress TensorFlow log messages
# Log level meanings
# Value	What WE see '0'	All logs (default), '1'	Hide INFO logs, '2'	Hide INFO and WARNING logs, '3'	Hide INFO, WARNING and ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# os is a built-in Python module, It allows Python to interact with the operating system
# environ is a dictionary-like object, It stores environment variables for the current process
# TF → TensorFlow
# CPP → C++ backend (TensorFlow core is written in C++)
# MIN_LOG_LEVEL → Minimum severity of logs to display

df = pd.read_csv("apple_stock.csv")
print(df.head())
print(df.info())
print(df.describe())


# Initial Data Visualization
# Plot 1 - Open and Close Prices of time 

plt.figure(figsize=(12,6), dpi=120)     # dpi → higher resolution
plt.plot(df['date'], df['open'], label='open', color='blue', alpha=0.7)
plt.plot(df['date'], df['close'], label='close', color='red', alpha=0.7)  # alpha → transparency (very important)
plt.title('open_close_price_overtime')
plt.legend()
# plt.show()  commented it out as we dont want again and again this visualization while performin task 


# Plot 2 - Trading Volume (check for outlier)
plt.figure(figsize=(12,6), dpi=120)
plt.plot(df['date'], df['volume'], label='volume', color='orange', alpha=0.7)
plt.title('stock_volume_overtime')
# plt.show()  commented it out as we dont want again and again this visualization while performin task 

# Drop non_numeric columns 
numeric_data = df.select_dtypes(include=["int64", "float64"])

# plot 3 - check for correlation between features 
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot= True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
# plt.show() commented it out as we dont want again and again this visualization while performin task 

# convert the Date into Date time then create a date filter 

df['date'] = pd.to_datetime(df['date'])

prediction = df.loc[
    (df['date']>datetime(2019,1,1)) & 
    (df['date']<datetime(2022,12,31))
]

plt.figure(figsize=(12,6))
plt.plot(df['date'], df['close'], color='blue')
plt.ylabel('close')
plt.xlabel('Date')
plt.title('Price_overtime')
# plt.show()


# Prepare for the LSTM Model (Sequential)
stock_close = df.filter(['close'])
dataset = stock_close.values  # converts to the numpy array 
training_data_len  = int(np.ceil(len(dataset) * .95))

# preprocessing stages 
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len]  # 95% of the all data

x_train, y_train = [], []

# create a sliding window for the stock(60 days)
for i in range(60, len(training_data)):
    x_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i,0])

# convert into array for tensorflow 
x_train, y_train = np.array(x_train), np.array(y_train)

# Lets take them to 3D array so tensorflow can interpret them better 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Build the Model 
model = keras.models.Sequential()

# First Layer(in this we add all our data)
model.add(keras.layers.LSTM(64, return_sequences = True, input_shape= (x_train.shape[1], 1)))
# Second Layer
model.add(keras.layers.LSTM(64, return_sequences = False))
# 3rd Layer (dense layer)
model.add(keras.layers.Dense(128, activation='relu'))
# 4th Layer
model.add(keras.layers.Dropout(0.5))
# Final Output Layer
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer='adam',
              loss='mae',
              metrics=[keras.metrics.RootMeanSquaredError()])


# Training Model 
training = model.fit(x_train, y_train, epochs=20, batch_size=32)

# prep the test data
test_data = scaled_data[training_data_len - 60:]
x_test, y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# convert x_test into numpy array to get prediction 
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

# Make a prediction
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# Plotting data 
train = df[:training_data_len]  # 95% of the data 
test = df[training_data_len:]   # remining 5%

test = test.copy()

test['predictions'] = predictions

plt.figure(figsize=(12,8))
plt.plot(train['date'],train['close'], label='Train(Actual)', color='blue')
plt.plot(test['date'], test['close'], label='Test (Actual)', color='orange')
plt.plot(test['date'], test['predictions'], label='predictions', color='red')
plt.title('Our_Stock_predictions')
plt.xlabel('Date')
plt.ylabel('close_price')
plt.legend()
plt.show()

