[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  

# recsys-rerank
ani dataset을 활용한 rerank 모델 구축 및 api 개발

## Dataset
I used [Anime Recommendation LTR](https://www.kaggle.com/datasets/ransakaravihara/anime-recommendation-ltr-dataset) in the Kaggle dataset.

## Results
### 1 stage
**LightGBM - LambdaRank**
| K  | Recall@K | MAP@K  | NDCG@K |
|----|----------|--------|--------|
| 3  |  0.0050  | 0.0204 | 0.0333 |
| 5  |  0.0100  | 0.0156 | 0.0387 |
| 10 |  0.0183  | 0.0094 | 0.0365 |
| 20 |  0.0350  | 0.0054 | 0.0352 |

**LightGBM - Binary**
| K  | Recall@K | MAP@K  | NDCG@K |
|----|----------|--------|--------|
| 3  |  0.0000  | 0.0000 | 0.0000 |
| 5  |  0.0050  | 0.0047 | 0.0141 |
| 10 |  0.0117  | 0.0036 | 0.0183 |
| 20 |  0.0367  | 0.0030 | 0.0300 |

**LightGBM - XE_NDCG**
| K  | Recall@K | MAP@K  | NDCG@K |
|----|----------|--------|--------|
| 3  |  0.0000  | 0.0000 | 0.0000 |
| 5  |  0.0050  | 0.0047 | 0.0141 |
| 10 |  0.0117  | 0.0036 | 0.0183 |
| 20 |  0.0367  | 0.0030 | 0.0300 |