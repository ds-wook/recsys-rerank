# %%
import numpy as np
import pandas as pd
import seaborn as sns

# %%
# Load the dataset
relavnce_scores = pd.read_csv("../input/anime-recommendation/relavence_scores.csv")
relavnce_scores.head()
# %%
sns.histplot(relavnce_scores["relavence_score"]);
# %%
