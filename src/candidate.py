from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from implicit.als import AlternatingLeastSquares
from omegaconf import DictConfig
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    # load raw dataset
    data = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.ratings}.csv")
    data["implicit_feedback"] = (data["rating"] >= 4).astype(int)

    # Create user-item interaction matrix
    interaction_matrix = data.pivot(index="userId", columns="movieId", values="implicit_feedback").fillna(0)

    # Train-test split
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # Convert train data to sparse matrix
    train_sparse = csr_matrix(interaction_matrix)

    # Step 3: Candidate Generation with ALS
    # Train ALS model
    als_model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=10)
    als_model.fit(train_sparse)

    # Generate top-k candidates for each user
    user_ids = interaction_matrix.index
    candidates = {user_id: als_model.recommend(user_id - 1, train_sparse[user_id - 1], N=100) for user_id in user_ids}

    # make candidates DataFrame
    candidate_data = []

    for user_id, items in tqdm(candidates.items()):
        for movie_id, score in zip(*items):
            label = int((train[(train["userId"] == user_id) & (train["movieId"] == movie_id)].shape[0]) > 0)
            candidate_data.append([user_id, movie_id, score, label])

    candidates_df = pd.DataFrame(candidate_data, columns=["userId", "movieId", "score", "label"])
    candidates_df.to_csv(Path(cfg.data.path) / f"train_{cfg.data.candidates}.csv", index=False)
    test.to_csv(Path(cfg.data.path) / "test.csv", index=False)


if __name__ == "__main__":
    _main()
