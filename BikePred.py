import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Import CSV File
df = pd.read_csv('https://raw.githubusercontent.com/Lorddhaval/Dataset/patch-1/Bike%20Prices.csv')

# Define dependent (y) and independent (X) variables
y = df['Selling_Price']
X = df[['Year', 'Seller_Type', 'Owner', 'KM_Driven', 'Ex_Showroom_Price']]

# Replace categorical values with numerical values
df.replace({'Seller_Type': {'Individual': 0, 'Dealer': 1}}, inplace=True)
df.replace({'Owner': {'1st owner': 0, '2nd owner': 1, '3rd owner': 2, '4th owner': 3}}, inplace=True)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=222529)

# Training Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Model Prediction
y_pred = lr.predict(X_test)

# Model Evaluation
mean_squared_error(y_test, y_pred)
mean_absolute_error(y_test, y_pred)
r2_score(y_test, y_pred)

# Visualization of Actual Vs Predicted Results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()

# Future Predictions
df_new = df.sample(1)
X_new = df_new.drop(['Brand', 'Model', 'Selling_Price'], axis=1)
y_pred_new = lr.predict(X_new)
y_pred_new
