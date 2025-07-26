import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Load Data & Models ------------------

# Collaborative Filtering (SVD)
df = pd.read_pickle("data.pkl")  # Your interaction data
svd_model = joblib.load("svd_model.pkl")  # Final saved SVD model

# Content-Based
content_vectorizer = joblib.load("final_content_vectorizer.pkl")
content_tfidf_matrix = joblib.load("final_content_tfidf_matrix.pkl")
content_products = pd.read_pickle("final_content_products.pkl")

# Prepare Product Info for display
product_info = df.drop_duplicates('product_id')[['product_id', 'category_code', 'brand']]
product_info['category_code'] = product_info['category_code'].fillna("Unknown Category")
product_info['brand'] = product_info['brand'].fillna("Unknown Brand")


def get_product_name(product_id):
    row = product_info[product_info['product_id'] == product_id]
    if not row.empty:
        return f"{row['brand'].values[0]} - {row['category_code'].values[0]}"
    return "Unknown Product"


# ------------------ Recommendation Functions ------------------

# 1) Collaborative Filtering (SVD)
def recommend_svd(user_id, n=5):
    all_products = df['product_id'].unique()
    user_products = df[df['user_id'] == user_id]['product_id'].unique()
    recommendations = []
    for product in all_products:
        if product not in user_products:
            pred = svd_model.predict(user_id, product)
            recommendations.append((product, pred.est))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]


# 2) Content-Based Filtering (Search by Similarity)
# ---------------- Manual Synonyms Dictionary ----------------
manual_synonyms = {
    "laptop": ["notebook", "ultrabook", "macbook", "chromebook"],
    "smartphone": ["mobile", "cellphone", "iphone", "android"],
    "headphones": ["earphones", "earbuds", "headset"],
    "tv": ["television", "smart tv", "oled tv", "led tv"],
    "camera": ["dslr", "mirrorless", "camcorder"]
}

def expand_with_manual_synonyms(keyword):
    """Expand keyword with manually defined synonyms."""
    expanded = [keyword]
    if keyword.lower() in manual_synonyms:
        expanded += manual_synonyms[keyword.lower()]
    return expanded

def recommend_content(keyword, n=5):
    expanded_keywords = expand_with_manual_synonyms(keyword)
    query = " ".join(expanded_keywords)
    
    query_vec = content_vectorizer.transform([query])  # use your loaded vectorizer
    sim_scores = cosine_similarity(query_vec, content_tfidf_matrix).flatten()
    
    top_indices = sim_scores.argsort()[::-1][:n]
    results = content_products.iloc[top_indices].copy()
    results["similarity_score"] = sim_scores[top_indices]
    return results


# 3) Hybrid (SVD + Content-Based Keyword Search)
def recommend_hybrid(user_id, keyword, n=5, alpha=0.7):
    # Get content-based similarity scores first
    query_vec = content_vectorizer.transform([keyword])
    sim_scores = cosine_similarity(query_vec, content_tfidf_matrix).flatten()

    all_products = content_products['product_id'].values
    hybrid_scores = []

    for i, pid in enumerate(all_products):
        # Collaborative Filtering prediction
        cf_score = svd_model.predict(user_id, pid).est
        # Content-Based similarity score
        cb_score = sim_scores[i]
        # Hybrid score = weighted sum
        score = alpha * cf_score + (1 - alpha) * cb_score
        hybrid_scores.append((pid, score))

    # Sort and return top N
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    results = []
    for pid, score in hybrid_scores[:n]:
        prod_row = content_products.loc[content_products['product_id'] == pid].iloc[0]
        results.append((pid, prod_row['brand'], prod_row['category_code'], score))
    return results


# ------------------ Streamlit UI ------------------

st.title("🛍️ Recommendation System")

option = st.selectbox(
    "Choose Recommendation Method:",
    ("Collaborative Filtering (SVD)", "Content-Based", "Hybrid")
)

# Dropdown for user selection (only for Collaborative & Hybrid)
user_ids = sorted(df['user_id'].unique())

if option == "Collaborative Filtering (SVD)":
    user_id = st.selectbox("Select User ID:", user_ids)
    if st.button("Get Recommendations"):
        results = recommend_svd(user_id)
        for pid, score in results:
            st.write(f"✅ {get_product_name(pid)} → Predicted Rating: {score:.2f}")

elif option == "Content-Based":
    keyword = st.text_input("Enter Keyword (e.g., phone, laptop):")
    if st.button("Search Products"):
        if keyword:
            results = recommend_content(keyword)
            for _, row in results.iterrows():
                st.write(f"✅ {row['brand']} - {row['category_code']} "
                         f"(Product ID: {row['product_id']}) → Similarity: {row['similarity_score']:.2f}")

else:  # Hybrid
    user_id = st.selectbox("Select User ID:", user_ids)
    keyword = st.text_input("Enter Keyword (for content similarity):")
    if st.button("Get Hybrid Recommendations"):
        if keyword:
            results = recommend_hybrid(user_id, keyword, 5)
            if results:
                for pid, brand, category, score in results:
                    st.write(f"✅ {brand} - {category} (Product ID: {pid}) → Hybrid Score: {score:.2f}")
            else:
                st.write("No matching recommendations found.")
