#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv('sale.csv')  # Make sure the file is in the same folder
df['date'] = pd.to_datetime(df['date'])


# In[5]:



# Add Date Features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.day_name()


# In[6]:


print(df.head())


# In[7]:


print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.describe())


# In[8]:


df.groupby('store')['item'].nunique()


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load and prepare data
df = pd.read_csv('sale.csv')  # Or the appropriate path
df['date'] = pd.to_datetime(df['date'])

# Filter data for Store 1, Item 1
df_one = df[(df['store'] == 1) & (df['item'] == 1)].copy()

# Set date as index and ensure daily frequency
df_one.set_index('date', inplace=True)
df_one = df_one.asfreq('D').sort_index()

# Split into training and testing (last 30 days for testing)
train = df_one[:-30]
test = df_one[-30:]

# Fit SARIMA model (basic parameters)
model = SARIMAX(train['sales'], order=(1,1,1), seasonal_order=(1,1,1,7))
model_fit = model.fit(disp=False)

# Forecast next 30 days
forecast = model_fit.forecast(steps=30)

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(train.index, train['sales'], label='Training Data')
plt.plot(test.index, test['sales'], label='Actual Sales (Last 30 Days)')
plt.plot(forecast.index, forecast, label='Forecasted Sales', linestyle='--')
plt.title("SARIMA Forecast – Store 1, Item 1")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[6]:


# Plot better zoomed-in view of last 1 year + forecast
plt.figure(figsize=(14, 6))

# Plot last 365 days of training data
plt.plot(train[-365:].index, train[-365:]['sales'], label='Recent Training Data', color='steelblue')

# Plot actual sales (last 30 days)
plt.plot(test.index, test['sales'], label='Actual Sales (Last 30 Days)', color='orange')

# Plot forecasted values
plt.plot(forecast.index, forecast, label='Forecasted Sales', color='green', linestyle='--', marker='o')

# Add a vertical line where forecast starts
plt.axvline(x=test.index[0], color='gray', linestyle=':', linewidth=1)

# Add titles and labels
plt.title('📈 SARIMA Forecast (Zoomed In) – Store 1, Item 1', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[7]:


import matplotlib.pyplot as plt

# Slice the last 90 days from training data
recent_train = train[-90:]

# Plot
plt.figure(figsize=(12, 5))
plt.plot(recent_train.index, recent_train['sales'], label='Recent Training Data', color='steelblue', linewidth=1)
plt.plot(test.index, test['sales'], label='Actual Sales (Last 30 Days)', color='orange', linewidth=2)
plt.plot(forecast.index, forecast, 'go--', label='Forecasted Sales', markersize=6)

# Vertical line to separate train and forecast
plt.axvline(test.index[0], color='gray', linestyle='dotted')

# Labels and legend
plt.title('SARIMA Forecast (Tightly Zoomed In) – Store 1, Item 1', fontsize=14)
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[8]:


# Fit SARIMA model on the entire available dataset
model_full = SARIMAX(df_one['sales'], order=(1,1,1), seasonal_order=(1,1,1,7))
model_fit_full = model_full.fit(disp=False)

# Forecast the next 30 days beyond the last known date
future_forecast = model_fit_full.forecast(steps=30)

# Create future date range
last_date = df_one.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(df_one.index[-90:], df_one['sales'][-90:], label='Recent Sales Data')  # last 90 days
plt.plot(future_dates, future_forecast, label='Forecasted Sales (Next 30 Days)', linestyle='--', marker='o', color='green')
plt.title("SARIMA Forecast – Future 30 Days (Store 1, Item 1)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[9]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(test['sales'], forecast)
rmse = np.sqrt(mean_squared_error(test['sales'], forecast))
mape = np.mean(np.abs((test['sales'] - forecast) / test['sales'])) * 100

print(f"📊 MAE: {mae:.2f}")
print(f"📊 RMSE: {rmse:.2f}")
print(f"📊 MAPE: {mape:.2f}%")


# In[9]:


df.columns


# In[10]:


df_pair = df[(df['store'] == 1) & (df['item'] == 1)].copy()


# In[12]:


# Filter data
df_pair = df[(df['store'] == 1) & (df['item'] == 1)].copy()

# Rename columns to Prophet format
df_pair = df_pair.rename(columns={'date': 'ds', 'sales': 'y'})

# Convert 'ds' to datetime
df_pair['ds'] = pd.to_datetime(df_pair['ds'])

df_pair.head()


# In[13]:


from prophet import Prophet

# Initialize the model
model = Prophet()

# Fit the model
model.fit(df_pair)


# In[14]:


# Create dataframe with future dates
future = model.make_future_dataframe(periods=90)

# View future dates
future.tail()


# In[15]:


# Make predictions
forecast = model.predict(future)

# Show some predictions
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[16]:


# Forecast plot
model.plot(forecast)


# In[17]:


model.plot_components(forecast)


# In[11]:


from prophet import Prophet

# Store results
forecasts = {}

# Loop through all items in Store 1
for item_id in df[df['store'] == 1]['item'].unique():
    print(f"Training Prophet for Store 1, Item {item_id}")
    
    # Filter data for this item
    df_pair = df[(df['store'] == 1) & (df['item'] == item_id)].copy()
    
    # Rename columns
    df_pair = df_pair.rename(columns={'date': 'ds', 'sales': 'y'})
    df_pair['ds'] = pd.to_datetime(df_pair['ds'])
    
    # Initialize and fit model
    model = Prophet()
    model.fit(df_pair)
    
    # Create future dates
    future = model.make_future_dataframe(periods=90)
    
    # Predict
    forecast = model.predict(future)
    
    # Save forecast and model
    forecasts[item_id] = {
        'model': model,
        'forecast': forecast
    }

    # (Optional) Plot forecast
    # model.plot(forecast)


# In[12]:


# For Item 3
forecast_item3 = forecasts[3]['forecast']
model_item3 = forecasts[3]['model']

# Show forecast
forecast_item3[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[13]:


import pandas as pd

# Collect all forecasts into a list
forecast_list = []

for item_id, data in forecasts.items():
    forecast_df = data['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_df['store'] = 1
    forecast_df['item'] = item_id
    forecast_list.append(forecast_df)

# Concatenate all forecasts
final_df = pd.concat(forecast_list)

# Save to CSV
final_df.to_csv("store1_items_forecast.csv", index=False)


# In[ ]:




