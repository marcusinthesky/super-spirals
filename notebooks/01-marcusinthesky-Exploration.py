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
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
from toolz.curried import *
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold


sys.path.append("../")
hv.extension("bokeh")

# %%
from super_spirals.neural_network import VAE

# %% [markdown]
# # Preliminary Analysis

# %%
X = load_iris()

# %%
pipeline = make_pipeline(
    StandardScaler(),
    VAE(
        hidden_layer_sizes=(5, 2), activation="tanh", divergence_weight=5, max_iter=500
    ),
)

# %%
pipeline.fit(X=X.data)

# %% [markdown]
# Inpect model

# %%
pipeline.named_steps["vae"].encoder.summary()

# %%
pipeline.named_steps["vae"].decoder.summary()

# %% [markdown]
# Analyze the reconstruced data, used for denoising

# %%
# original data
sample = X.data[:10, :]
sample

# %%
# denoised
pipe(sample, pipeline.transform, pipeline.inverse_transform)

# %% [markdown]
# Generate new samples

# %%
pipe(
    np.random.multivariate_normal(np.zeros(2), np.diag(np.ones(2)), size=10),
    pipeline.inverse_transform,
)

# %% [markdown]
# Analyze clusters

# %%
# %%output filename='../media/01-iris-latent' fig='png'
(
    pipe(
        X.data,
        pipeline.transform,
        partial(pd.DataFrame, columns=["Component 1", "Component 2"]),
    )
    .assign(label=X.target)
    .assign(label=lambda d: d.label.replace(dict(enumerate(X.target_names))))
    .hvplot.scatter(
        x="Component 1",
        y="Component 2",
        color="label",
        label="Variational Autoencoder Latent Encoding",
    )
)

# %%
