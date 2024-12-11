# %%
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split

# %%
# Step 1: Load MovieLens Dataset
# Download dataset: https://grouplens.org/datasets/movielens/
data = pd.read_csv("../input/ml-latest-small/ratings.csv")
data.head()
# %%
# Step 2: Preprocessing
# Filter implicit feedback (ratings >= 4 are considered as positive feedback)
data["implicit_feedback"] = (data["rating"] >= 4).astype(int)

# Create user-item interaction matrix
interaction_matrix = data.pivot(index="userId", columns="movieId", values="implicit_feedback").fillna(0)

# %%
# Train-test split
train, test = train_test_split(data, test_size=0.2, random_state=42)

# %%
# Convert train data to sparse matrix
train_sparse = csr_matrix(interaction_matrix)
# %%
# Step 3: Candidate Generation with ALS
# Train ALS model
als_model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=10)
als_model.fit(train_sparse)
# %%
# Generate top-k candidates for each user
user_ids = interaction_matrix.index
candidates = {user_id: als_model.recommend(user_id - 1, train_sparse[user_id - 1], N=100) for user_id in user_ids}
# %%
# Step 4: Reranking Model
# Prepare reranking training data
train_data = []
for user_id, items in candidates.items():
    for movie_id, score in zip(*items):
        label = int((train[(train["userId"] == user_id) & (train["movieId"] == movie_id)].shape[0]) > 0)
        train_data.append([user_id, movie_id, score, label])

rerank_train = pd.DataFrame(train_data, columns=["userId", "movieId", "score", "label"])
rerank_train.head()
# %%
rerank_train["label"].value_counts()
# %%
# Add features for reranking (here we use item_id, user_id, and ALS score as features)
user_features = train.groupby("userId")["movieId"].agg({"sum", "count"}).reset_index()
user_features.head()

# %%

# Train a rerank model (Gradient Boosting)
rerank_model = GradientBoostingClassifier()
rerank_model.fit(rerank_train[["user_id_feature", "item_id_feature", "score_feature"]], rerank_train["label"])

# Step 5: Evaluation
# Generate reranked lists for test users
ndcg_scores = []
for user_id in test["user_id"].unique():
    test_items = test[test["user_id"] == user_id]
    if user_id in candidates:
        candidate_items = candidates[user_id]
        candidate_item_ids = [item_id for item_id, _ in candidate_items]
        test_items = test_items[test_items["item_id"].isin(candidate_item_ids)]
        test_items["predicted_score"] = rerank_model.predict_proba(test_items[["user_id", "item_id"]])[:, 1]
        reranked_items = test_items.sort_values("predicted_score", ascending=False)["item_id"].tolist()
        true_relevance = test_items.sort_values("predicted_score", ascending=False)["implicit_feedback"].tolist()
        ndcg_scores.append(ndcg_score([true_relevance], [reranked_items]))

print("Average nDCG Score:", np.mean(ndcg_scores))
