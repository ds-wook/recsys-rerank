from __future__ import annotations

from pathlib import Path

import hydra
import lightgbm as lgb
from catboost import CatBoostRanker
from omegaconf import DictConfig
from prettytable import PrettyTable

from data import load_test_candidate
from evalution import map_at_k, ndcg_at_k, recall_at_k


@hydra.main(config_path="../config/", config_name="inference", version_base="1.3.1")
def _main(cfg: DictConfig):
    already_liked, candidates = load_test_candidate(cfg)

    ranker = (
        lgb.Booster(model_file=Path(cfg.models.model_path) / f"{cfg.models.results}.model")
        if cfg.models.name == "lightgbm"
        else CatBoostRanker().load_model(Path(cfg.models.model_path) / f"{cfg.models.results}.model")
    )

    # Make predictions
    preds = ranker.predict(candidates[cfg.stores.features])

    # Add predictions to the candidates DataFrame
    candidates["preds"] = preds
    candidates = candidates.sort_values(by=["userId", "preds"], ascending=[True, False])
    candidates = candidates.groupby("userId").head(20)

    # Evaluate the model
    candidates = candidates.groupby("userId")["movieId"].apply(list).reset_index()
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
