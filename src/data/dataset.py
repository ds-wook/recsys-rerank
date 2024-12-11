# This code is a Python file that defines a function for preprocessing and loading datasets.
# The function reads in the dataset, performs necessary preprocessing tasks,
# and transforms it into a format that can be used for model training.
from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def load_train_candidate(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the training dataset for candidate generation.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        tuple: Tuple of DataFrames (X_train, y_train).
    """
    # Load the training dataset
    train = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.train}.csv")
    movies = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.movies}.csv")

    genres_df = movies["genres"].str.get_dummies(sep="|")
    movies = pd.concat([movies, genres_df], axis=1)

    # merge the datasets
    train = train.merge(movies, on="movieId", how="left")

    # train-test split
    X = train.drop(columns=["label"])
    y = train["label"]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_valid, y_train, y_valid


def generate_candidates_test(user_ids, popular_movies, user_seen_movies):
    candidates = []
    for user_id in user_ids:
        seen_movies = user_seen_movies.get(user_id, set())
        candidate_movies = list(set(popular_movies) - seen_movies)
        for movie_id in candidate_movies:
            candidates.append({"userId": user_id, "movieId": movie_id})
    return pd.DataFrame(candidates)


def load_test_candidate(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the test dataset for candidate generation.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        tuple: Tuple of DataFrames (X_test, y_test).
    """
    # Load the test dataset
    test = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.test}.csv")
    already_liked_movies = test.groupby("userId")["movieId"].apply(list)

    ratings = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.ratings}.csv")

    # 평균 평점 상위 영화 추출 (전체 사용자의 인기 영화)
    popular_movies = (
        ratings.groupby("movieId")["rating"].mean().sort_values(ascending=False).head(cfg.top_k).index.tolist()
    )

    user_seen_movies = test.groupby("userId")["movieId"].apply(set).to_dict()
    candidates = generate_candidates_test(test["userId"].unique(), popular_movies, user_seen_movies)

    movies = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.movies}.csv")

    genres_df = movies["genres"].str.get_dummies(sep="|")
    movies = pd.concat([movies, genres_df], axis=1)

    # merge the datasets
    test = candidates.merge(movies, on="movieId")
    print(already_liked_movies)
    return already_liked_movies, test
