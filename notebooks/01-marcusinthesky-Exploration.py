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

# %% {"slideshow": {"slide_type": "skip"}}
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

# %% {"slideshow": {"slide_type": "skip"}}
from super_spirals.neural_network import VAE

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # Preliminary Analysis

# %% {"slideshow": {"slide_type": "skip"}}
X = load_iris()

# %% {"slideshow": {"slide_type": "skip"}}
pipeline = make_pipeline(
    StandardScaler(),
    VAE(
        hidden_layer_sizes=(5, 2), activation="tanh", divergence_weight=5, max_iter=500
    ),
)

# %% {"slideshow": {"slide_type": "skip"}}
pipeline.fit(X=X.data)

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# __Inpect model__

# %% [markdown] {"slideshow": {"slide_type": "-"}}
# Encoder

# %% {"slideshow": {"slide_type": "-"}}
pipeline.named_steps["vae"].encoder.summary()

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# __Decoder__

# %% {"slideshow": {"slide_type": "-"}}
pipeline.named_steps["vae"].decoder.summary()

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# __Denoising__

# %% [markdown] {"slideshow": {"slide_type": "-"}}
# Original data

# %% {"slideshow": {"slide_type": "-"}}
# original data
sample = X.data[:10, :]
pipe(sample, 
     partial(pd.DataFrame, columns=X.feature_names))

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# 'Denoised' Data

# %% {"slideshow": {"slide_type": "-"}}
# denoised
pipe(sample, 
     pipeline.transform, 
     pipeline.inverse_transform,
     partial(pd.DataFrame, columns=X.feature_names))

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# __Generate new samples__

# %% {"slideshow": {"slide_type": "-"}}
pipe(
    np.random.multivariate_normal(np.zeros(2), np.diag(np.ones(2)), size=10),
    pipeline.inverse_transform,
    
)

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# __Dimensionality Reduction__

# %% {"slideshow": {"slide_type": "-"}}
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

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# __Anomaly Detection__

# %% [markdown] {"slideshow": {"slide_type": "-"}}
# Assume normal, use Z-scores to filter outliers
