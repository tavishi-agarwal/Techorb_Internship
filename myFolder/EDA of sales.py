
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('sale.csv')  # Make sure the file is in the same folder
df['date'] = pd.to_datetime(df['date'])




print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.describe())





# Add Date Features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.day_name()



daily_sales = df.groupby('date')['sales'].sum().reset_index()

plt.figure(figsize=(14,5))
plt.plot(daily_sales['date'], daily_sales['sales'], label='Total Daily Sales')
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



item_sales = df.groupby('item')['sales'].sum().reset_index().sort_values(by='sales', ascending=False)

plt.figure(figsize=(12,5))
sns.barplot(x='item', y='sales', data=item_sales.head(10))
plt.title('Top 10 Best-Selling Items')
plt.xlabel('Item ID')
plt.ylabel('Total Sales')
plt.show()

plt.figure(figsize=(12,5))
sns.barplot(x='item', y='sales', data=item_sales.tail(10))
plt.title('Bottom 10 Least-Selling Items')
plt.xlabel('Item ID')
plt.ylabel('Total Sales')
plt.show()




monthly_sales = df.groupby(['year', 'month'])['sales'].sum().reset_index()
monthly_sales['year_month'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(day=1))

plt.figure(figsize=(14,5))
sns.lineplot(x='year_month', y='sales', data=monthly_sales)
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



store_sales = df.groupby('store')['sales'].sum().reset_index().sort_values(by='sales', ascending=False)

plt.figure(figsize=(12,5))
sns.barplot(x='store', y='sales', data=store_sales)
plt.title('Total Sales by Store')
plt.xlabel('Store ID')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()



monthly_store_sales = df.groupby(['store', 'year', 'month'])['sales'].sum().reset_index()
monthly_store_sales['year_month'] = pd.to_datetime(monthly_store_sales[['year', 'month']].assign(day=1))

plt.figure(figsize=(14,6))
for store_id in monthly_store_sales['store'].unique():
    subset = monthly_store_sales[monthly_store_sales['store'] == store_id]
    plt.plot(subset['year_month'], subset['sales'], label=f'Store {store_id}')
plt.title('Monthly Sales Trend per Store')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.show()




# 7. 🏆 Best & Worst Selling Item in Each Store


store_item_sales = df.groupby(['store', 'item'])['sales'].sum().reset_index()

# Best selling item per store
best_items = store_item_sales.loc[store_item_sales.groupby('store')['sales'].idxmax()].reset_index(drop=True)
best_items.rename(columns={'item': 'best_item', 'sales': 'best_sales'}, inplace=True)

# Worst selling item per store
worst_items = store_item_sales.loc[store_item_sales.groupby('store')['sales'].idxmin()].reset_index(drop=True)
worst_items.rename(columns={'item': 'worst_item', 'sales': 'worst_sales'}, inplace=True)


store_best_worst = pd.merge(best_items, worst_items, on='store')
print("📊 Best & Worst Selling Items per Store:")
print(store_best_worst)

plt.figure(figsize=(14,6))
sns.barplot(x='store', y='best_sales', data=store_best_worst, color='green', label='Best Item Sales')
sns.barplot(x='store', y='worst_sales', data=store_best_worst, color='red', label='Worst Item Sales')
plt.title('Best vs Worst Selling Item per Store')
plt.xlabel('Store')
plt.ylabel('Total Sales')
plt.legend()
plt.tight_layout()
plt.show()


# 🏆 Best & Worst Selling Item (by ID) in Each Store


store_item_sales = df.groupby(['store', 'item'])['sales'].sum().reset_index()

best_items = store_item_sales.loc[store_item_sales.groupby('store')['sales'].idxmax()].reset_index(drop=True)
best_items = best_items[['store', 'item']].rename(columns={'item': 'best_item'})


worst_items = store_item_sales.loc[store_item_sales.groupby('store')['sales'].idxmin()].reset_index(drop=True)
worst_items = worst_items[['store', 'item']].rename(columns={'item': 'worst_item'})

store_best_worst_items = pd.merge(best_items, worst_items, on='store')


print("🏬 Best and Worst Selling Item (ID only) per Store:")
print(store_best_worst_items)




#Which items consistently sell more or less 
df.groupby('item')['sales'].sum().sort_values(ascending=False)




df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek


df['dayofweek'] = df['date'].dt.dayofweek
# 0 = Monday, 6 = Sunday
dow_sales = df.groupby('dayofweek')['sales'].sum()

plt.figure(figsize=(10,5))
sns.barplot(x=dow_sales.index, y=dow_sales.values, palette='magma')
plt.title('Total Sales by Day of Week')
plt.xlabel('Day of the Week')
plt.ylabel('Total Sales')
plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.tight_layout()
plt.show()




# Group sales by month
monthly_sales = df.groupby('month')['sales'].sum().reset_index()

# Plot
plt.figure(figsize=(10,5))
sns.barplot(x='month', y='sales', data=monthly_sales, palette='viridis')

plt.title('Total Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(ticks=range(0, 12), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()





