import pandas as pd
from implicit.als import AlternatingLeastSquares
from omegaconf import DictConfig
from scipy.sparse import csr_matrix


class CandidateModel:
    def __init__(self, cfg: DictConfig):
        """
        Implicit-based recommender class for candidate generation.

        Args:
            factors (int): Number of latent factors.
            iterations (int): Number of ALS iterations.
            regularization (float): Regularization parameter.
            random_state (int): Random state for reproducibility.
        """
        self.cfg = cfg
        self.model = AlternatingLeastSquares(**self.cfg.model.params)
        self.user_item_matrix = None
        self.item_mapping = None
        self.user_mapping = None

    def fit(self, data: pd.DataFrame):
        """
        Fit the model using the input data.

        Args:
            data (pd.DataFrame): DataFrame with columns ['userId', 'movieId', 'rating'].
        """
        # Create user-item matrix
        user_item = data.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        self.user_item_matrix = csr_matrix(user_item.values)

        # Map user and item indices
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(user_item.index)}
        self.item_mapping = {movie_id: idx for idx, movie_id in enumerate(user_item.columns)}

        # Train the model (requires transposed matrix: item-user)
        self.model.fit(self.user_item_matrix.T)

    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        """
        Generate candidate recommendations for a given user.

        Args:
            user_id (int): User ID for recommendation.
            top_k (int): Number of candidates to recommend.

        Returns:
            list: List of recommended item IDs.
        """
        if self.user_item_matrix is None:
            raise ValueError("Model is not fitted. Call `fit` before recommending.")

        # Convert user ID to index
        user_idx = self.user_mapping.get(user_id)
        if user_idx is None:
            raise ValueError(f"User ID {user_id} not found in the training data.")

        # Generate recommendations
        recommended = self.model.recommend(
            user_idx,
            self.user_item_matrix[user_idx],
            N=top_k,
            filter_already_liked_items=True,
        )
        # Map item indices back to item IDs
        return [item for item, _ in recommended]
