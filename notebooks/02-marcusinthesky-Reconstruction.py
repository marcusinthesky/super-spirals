# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import sys
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    make_multilabel_classification,
)
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
from typing import Dict
from toolz.curried import *
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


sys.path.append("../")
hv.extension("bokeh")

# %%
from super_spirals.metrics import reconstruction_benchmark
from super_spirals.neural_network import VAE


# %%
def get_models():
    return {
        "VAE (relu)": make_pipeline(
            StandardScaler(),
            VAE(activation="relu", max_iter=300, hidden_layer_sizes=(4, 2)),
        ),
        "VAE (tanh)": make_pipeline(
            StandardScaler(),
            VAE(activation="tanh", max_iter=300, hidden_layer_sizes=(4, 2)),
        ),
        "PCA": make_pipeline(StandardScaler(), PCA(n_components=2)),
        "ICA": make_pipeline(StandardScaler(), FastICA(n_components=2)),
        "KPCA (rbf)": make_pipeline(
            StandardScaler(),
            KernelPCA(n_components=2, kernel="rbf", fit_inverse_transform=True),
        ),
        "KPCA (cosine)": make_pipeline(
            StandardScaler(),
            KernelPCA(n_components=2, kernel="cosine", fit_inverse_transform=True),
        ),
    }


# %%
iris_models = get_models()
iris_df, iris_reconstruction, iris_silhouette, iris_plot = reconstruction_benchmark(
    load_iris(), iris_models, "Iris Dataset"
)

# %%
# %%output filename='../media/02-iris-loss' fig='png'
iris_reconstruction

# %%
# %%output filename='../media/02-iris-silhouette' fig='png'
iris_silhouette

# %%
# %%output filename='../media/02-iris-latent' fig='png'
iris_plot

# %%
wine_models = get_models()
wine_df, wine_reconstruction, wine_silhouette, wine_plot = reconstruction_benchmark(
    load_wine(), wine_models, "Wine Dataset"
)

# %%
# %%output filename='../media/02-wine-loss' fig='png'
wine_reconstruction

# %%
# %%output filename='../media/02-wine-silhouette' fig='png'
wine_silhouette

# %%
# %%output filename='../media/02-wine-latent' fig='png'
wine_plot

# %%
cancer_models = get_models()
cancer_df, cancer_reconstruction, cancer_silhouette, cancer_plot = reconstruction_benchmark(
    load_breast_cancer(), cancer_models, "Cancer Dataset"
)

# %%
# %%output filename='../media/02-cancer-loss' fig='png'
cancer_reconstruction

# %%
# %%output filename='../media/02-cancer-silhouette' fig='png'
cancer_silhouette

# %%
# %%output filename='../media/02-cancer-latent' fig='png'
cancer_plot


# %%
class get_random:
    random = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=5)
    data = random[0]
    target = pd.DataFrame(random[1]).idxmax(1).to_numpy()


# %%
random_models = get_models()
random_df, random_reconstruction, random_silhouette, random_plot = reconstruction_benchmark(
    get_random(), random_models, "Random Dataset"
)

# %%
# %%output filename='../media/02-random-loss' fig='png'
random_reconstruction

# %%
# %%output filename='../media/02-random-silhouette' fig='png'
random_silhouette

# %%
# %%output filename='../media/02-random-latent' fig='png'
random_plot

# %%
