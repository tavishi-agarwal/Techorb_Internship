#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import joblib

# --- Load Data ---
df = pd.read_csv("2019-Oct.csv", nrows=100000)

# Map event_type to ratings
rating_map = {'purchase': 5, 'cart': 4, 'view': 3}
df['rating'] = df['event_type'].map(rating_map)
df = df.drop(columns=['event_type'])

# --- Prepare Data for Surprise ---
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# --- Train the Model ---
model = SVD()
model.fit(trainset)

# --- Evaluate the Model ---
predictions = model.test(testset)
print("RMSE:", accuracy.rmse(predictions))

# --- Create Product Info Table ---
product_info = df.drop_duplicates('product_id')[['product_id', 'category_code', 'brand']]
product_info['category_code'] = product_info['category_code'].fillna("Unknown Category")
product_info['brand'] = product_info['brand'].fillna("Unknown Brand")

# Function to get product name
def get_product_name(product_id):
    row = product_info[product_info['product_id'] == product_id]
    if not row.empty:
        category = row['category_code'].values[0]
        brand = row['brand'].values[0]
        return f"{brand} - {category}"
    return "Unknown Product"

# --- Recommend Top-N Products ---
def recommend_products(user_id, n=5):
    all_products = df['product_id'].unique()
    user_products = df[df['user_id'] == user_id]['product_id'].unique()

    recommendations = []
    for product in all_products:
        if product not in user_products:
            pred = model.predict(user_id, product)
            recommendations.append((product, pred.est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]

# Example usage
sample_user = df['user_id'].iloc[0]
print(f"Top 5 Recommendations for User {sample_user}:")
for pid, score in recommend_products(sample_user, 5):
    print(f"{get_product_name(pid)} (Product ID: {pid}) -> Predicted Rating: {score:.2f}")

# --- Save Model & Data ---
joblib.dump(model, "svd_model.pkl")
df.to_pickle("data.pkl")

print("✅ Model and data saved successfully!")
