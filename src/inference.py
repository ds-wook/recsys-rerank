from __future__ import annotations

from pathlib import Path

import hydra
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from omegaconf import DictConfig
from prettytable import PrettyTable
from tqdm import tqdm

from data import load_dataset, load_test_dataset
from evalution import map_at_k, ndcg_at_k, recall_at_k
from model import TreeModel


def candidate_generation(
    user_id: int, candidate_pool: list[str], user_to_anime_map: dict[int, list[str]], N: int
) -> tuple[list[str], list[str]]:
    already_interacted = user_to_anime_map[user_id]
    candidates = list(set(candidate_pool) - set(already_interacted))
    return already_interacted, np.random.choice(candidates, size=N)


def generate_predictions(
    cfg: DictConfig,
    user_id: int,
    user_2_anime_map: dict[int, list[str]],
    candidate_pool: list[str],
    feature_columns: list[str],
    anime_id_2_name_map: dict[int, list[str]],
    ranker: TreeModel,
    N: int = 100,
) -> pd.DataFrame:
    anime_info_df_final, _, user_info = load_dataset(cfg)
    already_liked, candidates = candidate_generation(user_id, candidate_pool, user_2_anime_map, N=10000)
    candidates_df = pd.DataFrame(data=pd.Series(candidates, name="anime_id"))
    features = anime_info_df_final.merge(candidates_df)
    features["user_id"] = user_id
    features = features.merge(user_info)

    already_liked = list(already_liked)
    if len(already_liked) < len(candidates):
        append_list = np.full(fill_value=-1, shape=(len(candidates) - len(already_liked)))
        already_liked.extend(list(append_list))

    predictions = pd.DataFrame(index=candidates)
    predictions["already_liked"] = [anime_id_2_name_map.get(_id) for _id in already_liked[0 : len(predictions)]]
    predictions["name"] = np.array([anime_id_2_name_map.get(id_) for id_ in candidates])
    predictions["score"] = ranker.predict(features[feature_columns])
    predictions = predictions.sort_values(by="score", ascending=False).head(N)

    return predictions


@hydra.main(config_path="../config/", config_name="inference", version_base="1.3.1")
def _main(cfg: DictConfig):
    user_2_anime_map, candidate_pool, anime_id_2_name_map = load_test_dataset(cfg)

    ranker = (
        lgb.Booster(model_file=Path(cfg.models.model_path) / f"{cfg.models.results}.model")
        if cfg.models.name == "lightgbm"
        else CatBoostRanker().load_model(Path(cfg.models.model_path) / f"{cfg.models.results}.model")
    )

    recommendations = pd.DataFrame()
    user_sample = np.random.choice(list(user_2_anime_map.keys()), size=10, replace=False)

    for user_id in tqdm(user_sample):
        predictions = generate_predictions(
            cfg=cfg,
            user_id=user_id,
            user_2_anime_map=user_2_anime_map,
            candidate_pool=candidate_pool,
            feature_columns=cfg.stores.features,
            anime_id_2_name_map=anime_id_2_name_map,
            ranker=ranker,
            N=cfg.N,
        )
        predictions["user_id"] = user_id
        recommendations = pd.concat([recommendations, predictions])

    # Print evaluation metrics
    already_liked = recommendations.groupby("user_id")["already_liked"].apply(list)
    candidates = recommendations.groupby("user_id")["name"].apply(list)

    table = PrettyTable()
    table.field_names = ["K", "Recall@K", "MAP@K", "NDCG@K"]

    for k in [3, 5, 10, 20]:
        recall_at_k_score = recall_at_k(already_liked, candidates, k=k)
        map_at_k_score = map_at_k(already_liked, candidates, k=k)
        ndcg_at_k_score = ndcg_at_k(already_liked, candidates, k=k)

        table.add_row([k, f"{recall_at_k_score: .4f}", f"{map_at_k_score:.4f}", f"{ndcg_at_k_score:.4f}"])

    print(table)


if __name__ == "__main__":
    _main()
