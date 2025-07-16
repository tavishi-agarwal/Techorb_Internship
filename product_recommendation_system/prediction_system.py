import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load the saved model and data
# -------------------------------
model = joblib.load("C:/Users/delis/svd_model.pkl")
df = pd.read_pickle("C:/Users/delis/data.pkl")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Product Recommendation System", page_icon="🛒", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: white;'>🛒 Product Recommendation System</h1>
    """,
    unsafe_allow_html=True
)

# Get unique users
unique_users = df["user_id"].unique()

selected_user = st.selectbox("Select a User:", unique_users)

if st.button("Get Recommendations"):
    # Get all products
    all_products = df["product_id"].unique()

    # Predict ratings for all products
    predictions = []
    for product_id in all_products:
        pred = model.predict(selected_user, product_id)
        predictions.append((product_id, pred.est))

    # Sort by highest predicted rating
    top_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

    st.markdown("### 🔥 Top Recommendations for You:")

    for product_id, rating in top_recommendations:
        # Get product details
        product_row = df[df["product_id"] == product_id]
        if not product_row.empty:
            product_info = product_row.iloc[0]
            brand = product_info["brand"] if pd.notna(product_info["brand"]) else "Unknown Brand"

            # Generate a readable product name from category_code
            if pd.notna(product_info["category_code"]):
                product_name = product_info["category_code"].split(".")[-1].capitalize()
            else:
                product_name = f"Product {product_id}"

            st.write(f"**{product_name}** ({brand}) → Predicted Rating: {rating:.2f}")
        else:
            st.write(f"Product ID: {product_id} → Predicted Rating: {rating:.2f}")
