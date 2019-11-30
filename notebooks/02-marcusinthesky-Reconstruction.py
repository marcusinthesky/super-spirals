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

# %% {"slideshow": {"slide_type": "skip"}, "language": "html"}
# <style>
# div.input {
#     display:none;
# }
# </style>

# %% {"slideshow": {"slide_type": "skip"}, "language": "html"}
# <style>
# div.input {
#     display:contents;
# }
# </style>

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # Dimensionality Reduction
# 1. Compressed representation which capture all relevant structure in the data
# 2. Representation useful in supervised and unsupervised learning tasks

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Models
# 1. VAE (Tanh)
# 2. VAE (ReLU)
# 3. PCA
# 4. ICA
# 5. Kernel PCA (RBF)
# 6. Kernel PCA (Cosine)

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Data
# 1. Fisher's Iris data
# 2. UCI ML Wine Data
# 3. UCI ML Breast Cancer Wisconsin (Diagnostic)
#

# %% {"slideshow": {"slide_type": "skip"}}
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

# %% {"slideshow": {"slide_type": "skip"}}
from super_spirals.reports import reconstruction_benchmark
from super_spirals.neural_network import VAE


# %% {"slideshow": {"slide_type": "skip"}}
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


# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # Iris

# %% {"slideshow": {"slide_type": "skip"}}
iris_models = get_models()
iris_df, iris_reconstruction, iris_silhouette, iris_plot = reconstruction_benchmark(
    load_iris(), iris_models, "Iris Dataset"
)

# %% {"slideshow": {"slide_type": "slide"}}
# %%output filename='../media/02-iris-loss' fig='png'
iris_reconstruction

# %% {"slideshow": {"slide_type": "slide"}}
# %%output filename='../media/02-iris-silhouette' fig='png'
iris_silhouette

# %% {"slideshow": {"slide_type": "slide"}}
# %%output filename='../media/02-iris-latent' fig='png'
iris_plot

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # Wine

# %% {"slideshow": {"slide_type": "skip"}}
wine_models = get_models()
wine_df, wine_reconstruction, wine_silhouette, wine_plot = reconstruction_benchmark(
    load_wine(), wine_models, "Wine Dataset"
)

# %% {"slideshow": {"slide_type": "slide"}}
# %%output filename='../media/02-wine-loss' fig='png'
wine_reconstruction

# %% {"slideshow": {"slide_type": "slide"}}
# %%output filename='../media/02-wine-silhouette' fig='png'
wine_silhouette

# %% {"slideshow": {"slide_type": "slide"}}
# %%output filename='../media/02-wine-latent' fig='png'
wine_plot

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # Cancer

# %% {"slideshow": {"slide_type": "skip"}}
cancer_models = get_models()
cancer_df, cancer_reconstruction, cancer_silhouette, cancer_plot = reconstruction_benchmark(
    load_breast_cancer(), cancer_models, "Cancer Dataset"
)

# %% {"slideshow": {"slide_type": "slide"}}
# %%output filename='../media/02-cancer-loss' fig='png'
cancer_reconstruction

# %% {"slideshow": {"slide_type": "slide"}}
# %%output filename='../media/02-cancer-silhouette' fig='png'
cancer_silhouette

# %% {"slideshow": {"slide_type": "slide"}}
# %%output filename='../media/02-cancer-latent' fig='png'
cancer_plot


# %% {"slideshow": {"slide_type": "skip"}}
class get_random:
    random = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=5)
    data = random[0]
    target = pd.DataFrame(random[1]).idxmax(1).to_numpy()


# %% {"slideshow": {"slide_type": "skip"}}
random_models = get_models()
random_df, random_reconstruction, random_silhouette, random_plot = reconstruction_benchmark(
    get_random(), random_models, "Random Dataset"
)

# %% {"slideshow": {"slide_type": "skip"}}
# %%output filename='../media/02-random-loss' fig='png'
random_reconstruction

# %% {"slideshow": {"slide_type": "skip"}}
# %%output filename='../media/02-random-silhouette' fig='png'
random_silhouette

# %% {"slideshow": {"slide_type": "skip"}}
# %%output filename='../media/02-random-latent' fig='png'
random_plot

# %% {"slideshow": {"slide_type": "skip"}}
