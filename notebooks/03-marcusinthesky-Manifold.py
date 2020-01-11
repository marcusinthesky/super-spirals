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

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # Manifold Learning
# 1. Data Exists in Manifold within the high dimensional space
# 2. Often non-linear surface

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Models
# - VAE (relu)
# - VAE (tanh)
# - PCA
# - LLE
# - Isomap
# - MDS
# - SpectralEmbedding
# - TSNE

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Data
# - S-curve
# - Swiss Roll

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

# %% {"slideshow": {"slide_type": "skip"}}
import sys
from sklearn.datasets import samples_generator
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
from toolz.curried import pipe, map, compose_left, partial
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold


sys.path.append("../")
hv.extension("bokeh")

# %% {"slideshow": {"slide_type": "skip"}}
from super_spirals.neural_network import LikelihoodVAE

# %% {"slideshow": {"slide_type": "skip"}}
n_points = 1000


# %% {"slideshow": {"slide_type": "skip"}}
def get_models():
    return {
        "VAE (relu)": make_pipeline(
            StandardScaler(),
            LikelihoodVAE(activation="relu", n_iter=300, hidden_layer_sizes=(4, 5, 2)),
        ),
        "VAE (tanh)": make_pipeline(
            StandardScaler(),
            LikelihoodVAE(activation="tanh", n_iter=300, hidden_layer_sizes=(4, 5, 2)),
        ),
        "PCA": make_pipeline(StandardScaler(), PCA(n_components=2)),
        "LLE": make_pipeline(
            StandardScaler(), manifold.LocallyLinearEmbedding(n_components=2)
        ),
        "Isomap": make_pipeline(StandardScaler(), manifold.Isomap(n_components=2)),
        "MDS": make_pipeline(StandardScaler(), manifold.MDS(n_components=2)),
        "SpectralEmbedding": make_pipeline(
            StandardScaler(), manifold.SpectralEmbedding(n_components=2)
        ),
        "TSNE": make_pipeline(StandardScaler(), manifold.TSNE(n_components=2)),
    }


# %% {"slideshow": {"slide_type": "skip"}}
def get_components(model, X, y, tag):

    latent = pipe(
        X,
        model.fit_transform,
        StandardScaler().fit_transform,
        PCA(whiten=True).fit_transform,
        partial(pd.DataFrame, columns=["Component 1", "Component 2"]),
    )

    return latent.assign(y=y).assign(tag=tag)


# %% {"slideshow": {"slide_type": "skip"}}
s_curve_models = get_models()
s_curve_X, s_curve_color = samples_generator.make_s_curve(n_points, random_state=0)

s_curve_components = pd.concat(
    [get_components(m, s_curve_X, s_curve_color, t) for t, m in s_curve_models.items()]
)

# %% {"slideshow": {"slide_type": "slide"}}
# %%output filename='../media/03-scurve-latent' fig='png'
(
    s_curve_components.hvplot.scatter(
        x="Component 1", y="Component 2", color="y", groupby="tag", cmap="spectral"
    )
    .layout()
    .opts(title="S-Curve Manifold", shared_axes=False)
    .cols(2)
)

# %% {"slideshow": {"slide_type": "skip"}}
swissroll_models = get_models()
swissroll_X, swissroll_color = samples_generator.make_swiss_roll(
    n_points, random_state=0
)

swissroll_components = pd.concat(
    [
        get_components(m, swissroll_X, swissroll_color, t)
        for t, m in swissroll_models.items()
    ]
)

# %% {"slideshow": {"slide_type": "slide"}}
# %%output filename='../media/03-swissroll-latent' fig='png'
(
    swissroll_components.hvplot.scatter(
        x="Component 1", y="Component 2", color="y", groupby="tag", cmap="spectral"
    )
    .layout()
    .opts(title="Swill-roll Manifold", shared_axes=False)
    .cols(2)
)
