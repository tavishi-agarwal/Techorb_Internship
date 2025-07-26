


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

vectorizer = joblib.load("final_content_vectorizer.pkl")
tfidf_matrix = joblib.load("final_content_tfidf_matrix.pkl")
products = pd.read_pickle("final_content_products.pkl")

svd_model = joblib.load("svd_model.pkl")  # Collaborative filtering model

def collaborative_recommend(user_id, top_n=20):
    """
    Returns top_n products the user hasn't interacted with, based on predicted ratings.
    """
    all_products = products['product_id'].unique()
    predictions = []

    for pid in all_products:
        est = svd_model.predict(user_id, pid).est
        predictions.append((pid, est))

    df = pd.DataFrame(predictions, columns=["product_id", "collab_score"])
    df = df.sort_values(by="collab_score", ascending=False).head(top_n)
    return df
#hybrid model
def hybrid_recommend(user_id, search_keyword=None, top_n=5, alpha=0.7):
    """
    Combines collaborative & content-based scores.
    alpha = weight for collaborative (0.7 by default)
    """
    # 1) Collaborative candidates
    collab_candidates = collaborative_recommend(user_id, top_n=30)
    candidate_indices = products[products['product_id'].isin(collab_candidates['product_id'])].index

    # 2) Content-based adjustment 
    if search_keyword:
        keyword_vector = vectorizer.transform([search_keyword])
        content_scores = cosine_similarity(keyword_vector, tfidf_matrix[candidate_indices]).flatten()
    else:
        content_scores = np.zeros(len(candidate_indices))

    # 3) Combine scores
    collab_scores = collab_candidates['collab_score'].values
    final_scores = alpha * collab_scores + (1 - alpha) * content_scores

    collab_candidates["hybrid_score"] = final_scores
    collab_candidates = collab_candidates.merge(products, on="product_id", how="left")

    return collab_candidates.sort_values(by="hybrid_score", ascending=False).head(top_n)

# ---------------- Example ----------------
user_id = 123  
keyword = "laptop"  
print(hybrid_recommend(user_id, search_keyword=keyword, top_n=5))





