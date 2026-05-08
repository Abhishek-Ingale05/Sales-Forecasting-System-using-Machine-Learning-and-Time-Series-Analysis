# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# Load Dataset
df = pd.read_csv("sales_forecasting_dataset.csv")

# Convert Date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Aggregate daily sales
sales = df.groupby('Date')['Units_Sold'].sum()

# Plot Sales Trend
plt.figure()
plt.plot(sales)
plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.show()

# Train-Test Split
train = sales[:150]
test = sales[150:]

#  ARIMA Model (Main Model)

model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))

# Plot Forecast
plt.figure()
plt.plot(train, label="Train")
plt.plot(test, label="Actual")
plt.plot(forecast, label="Forecast")
plt.legend()
plt.title("Sales Forecast vs Actual")
plt.show()

# Evaluation
mae = mean_absolute_error(test, forecast)
print("MAE:", mae)


#  Simple ML Model (Extra)

df['Day'] = range(len(df))

X = df[['Day']]
y = df['Units_Sold']

model_lr = LinearRegression()
model_lr.fit(X, y)

future_days = pd.DataFrame({'Day': range(len(df), len(df)+10)})
pred = model_lr.predict(future_days)

print("Next 10 Days Sales Prediction:")
print(pred)