# %%
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# %%
# Step 1: Load MovieLens Dataset
# Download dataset: https://grouplens.org/datasets/movielens/
data = pd.read_csv("../input/ml-latest-small/ratings.csv")
data.head()
# %%
# Step 2: Preprocessing
# Filter implicit feedback (ratings >= 4 are considered as positive feedback)
data["label"] = (data["rating"] >= 4).astype(int)

# Create user-item interaction matrix
interaction_matrix = data.pivot(index="userId", columns="movieId", values="label").fillna(0)

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

candidates_df = pd.DataFrame(train_data, columns=["userId", "movieId", "score", "label"])
candidates_df.head()
# %%
candidates_df["label"].value_counts()
# %%
candidates_df.shape
# %%