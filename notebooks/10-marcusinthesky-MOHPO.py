#%%
from time import time

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV

from super_spirals.model_selection import data_envelopment_analysis

# get some data
X, y = load_digits(return_X_y=True)
# build a classifier
clf = SGDClassifier(loss="hinge", penalty="elasticnet", fit_intercept=True)
# specify parameters and distributions to sample from
param_dist = {
    "average": [True, False],
    "l1_ratio": stats.uniform(0, 1),
    "alpha": stats.uniform(1e-4, 1e0),
}
# run randomized search
n_iter_search = 3
random_search = RandomizedSearchCV(
    clf, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1
)
random_search.fit(X, y)
cv_results_df = pd.DataFrame(random_search.cv_results_)


# %%
metrics = [
    "mean_fit_time",
    "mean_score_time",
    "mean_test_score",
]
metrics_greater_is_better = [False, False, True]
data_envelopment_analysis(
    validation_metrics=cv_results_df[metrics],
    greater_is_better=metrics_greater_is_better,
)


# %%
