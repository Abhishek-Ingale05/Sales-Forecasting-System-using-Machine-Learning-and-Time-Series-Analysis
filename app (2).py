# =====================================================
# SALES FORECASTING SYSTEM
# Using Machine Learning and Time Series Analysis
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("sales_forecasting_dataset.csv")

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date')

# =====================================================
# SALES TREND
# =====================================================

sales = df.groupby('Date')['Units_Sold'].sum()

plt.figure(figsize=(10,5))

plt.plot(sales)

plt.title("Sales Trend Over Time")

plt.xlabel("Date")

plt.ylabel("Units Sold")

plt.grid(True)

plt.show()

# =====================================================
# MOVING AVERAGE
# =====================================================

moving_avg = sales.rolling(window=7).mean()

plt.figure(figsize=(10,5))

plt.plot(
    sales,
    label="Daily Sales"
)

plt.plot(
    moving_avg,
    label="7-Day Moving Average",
    linewidth=3
)

plt.title("Sales Trend with Moving Average")

plt.xlabel("Date")

plt.ylabel("Units Sold")

plt.legend()

plt.grid(True)

plt.show()

# =====================================================
# MONTHLY SALES ANALYSIS
# =====================================================

monthly_sales = df.groupby(
    df['Date'].dt.month
)['Units_Sold'].sum()

plt.figure(figsize=(8,5))

monthly_sales.plot(
    kind='bar'
)

plt.title("Monthly Sales Analysis")

plt.xlabel("Month")

plt.ylabel("Units Sold")

plt.show()

# =====================================================
# MACHINE LEARNING MODEL
# LINEAR REGRESSION
# =====================================================

df['Day_Number'] = range(len(df))

X = df[['Day_Number']]

y = df['Units_Sold']

lr_model = LinearRegression()

lr_model.fit(X, y)

predictions = lr_model.predict(X)

# =====================================================
# LINEAR REGRESSION EVALUATION
# =====================================================

mae = mean_absolute_error(
    y,
    predictions
)

print("\nLinear Regression MAE:", round(mae,2))

# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    y.values,
    label="Actual Sales"
)

plt.plot(
    predictions,
    label="Predicted Sales"
)

plt.title(
    "Actual vs Predicted Sales"
)

plt.xlabel("Records")

plt.ylabel("Units Sold")

plt.legend()

plt.show()

# =====================================================
# TIME SERIES MODEL
# ARIMA
# =====================================================

train = sales[:150]

test = sales[150:]

arima_model = ARIMA(
    train,
    order=(2,1,2)
)

arima_fit = arima_model.fit()

forecast = arima_fit.forecast(
    steps=len(test)
)

# =====================================================
# FORECAST VS ACTUAL
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    train,
    label="Train Data"
)

plt.plot(
    test,
    label="Actual Sales"
)

plt.plot(
    forecast,
    label="Forecast Sales"
)

plt.title(
    "Forecast vs Actual Sales"
)

plt.xlabel("Date")

plt.ylabel("Units Sold")

plt.legend()

plt.show()

# =====================================================
# SALES GROWTH ANALYSIS
# =====================================================

growth = sales.pct_change() * 100

plt.figure(figsize=(10,5))

plt.plot(growth)

plt.title(
    "Sales Growth Rate (%)"
)

plt.xlabel("Date")

plt.ylabel("Growth %")

plt.grid(True)

plt.show()

# =====================================================
# FUTURE SALES FORECAST
# =====================================================

future_forecast = arima_fit.forecast(
    steps=10
)

print("\nNext 10 Days Sales Forecast")

for i, value in enumerate(
    future_forecast,
    start=1
):
    print(
        f"Day {i}: {round(value)} Units"
    )

# =====================================================
# INVENTORY RECOMMENDATION
# =====================================================

avg_sales = future_forecast.mean()

safety_stock = avg_sales * 0.20

recommended_inventory = (
    avg_sales + safety_stock
)

print("\nInventory Recommendation")

print(
    "Predicted Average Sales:",
    round(avg_sales)
)

print(
    "Safety Stock:",
    round(safety_stock)
)

print(
    "Recommended Inventory:",
    round(recommended_inventory)
)

# =====================================================
# BUSINESS INSIGHT REPORT
# =====================================================

best_month = monthly_sales.idxmax()

highest_sales = monthly_sales.max()

print("\nBUSINESS INSIGHT REPORT")

print(
    "Best Performing Month:",
    best_month
)

print(
    "Highest Monthly Sales:",
    highest_sales
)

print(
    "Average Daily Sales:",
    round(sales.mean())
)

print(
    "Maximum Daily Sales:",
    sales.max()
)

print(
    "Minimum Daily Sales:",
    sales.min()
)

# =====================================================
# END OF PROJECT
# =====================================================
