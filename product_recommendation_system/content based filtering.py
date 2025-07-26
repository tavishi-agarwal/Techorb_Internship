


import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Cleaned Products
products = pd.read_pickle("cleaned_products.pkl")
print("Loaded Products:", products.shape)

# 2. TF-IDF Vectorization

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(products['features'])



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

def recommend_content_manual(keyword, n=5):
    expanded_keywords = expand_with_manual_synonyms(keyword)
    print(f"\n🔍 Searching for: {expanded_keywords}")
    
    query = " ".join(expanded_keywords)
    
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    related_docs_indices = cosine_similarities.argsort()[::-1]
    related_docs_indices = [i for i in related_docs_indices if i < len(products)][:n]
    
    results = products.iloc[related_docs_indices].copy()
    results["similarity_score"] = cosine_similarities[related_docs_indices]
    return results





import joblib

# Save vectorizer & TF-IDF matrix
joblib.dump(vectorizer, "final_content_vectorizer.pkl")
joblib.dump(tfidf_matrix, "final_content_tfidf_matrix.pkl")

# Save cleaned products dataframe
products.to_pickle("final_content_products.pkl")

print("✅ Final content-based recommendation model saved!")







