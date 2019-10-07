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

# %% [markdown] {"slideshow": {"slide_type": "skip"}}
# # Sampling

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# __Aim__
# Truncated Bivariate Gaussian $\rightarrow$ 1D Representation $\rightarrow$ Truncated Bivariate Gaussian

# %% {"slideshow": {"slide_type": "skip"}}
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import holoviews as hv
import hvplot.pandas
from toolz.curried import *


sys.path.append("../")
hv.extension("bokeh")

# %% {"slideshow": {"slide_type": "skip"}}
from super_spirals.neural_network import VAE

# %% {"slideshow": {"slide_type": "skip"}}
X = np.random.multivariate_normal(
    mean=np.zeros(2), cov=np.diag(np.ones(2)), size=100000
)

beta = np.random.uniform(-0.1, 0.1, size=(2, 2))
data = pd.DataFrame(X @ beta).where(lambda d: d > 0).dropna(how="any").to_numpy()

# %% {"slideshow": {"slide_type": "skip"}}
vae = VAE(
    hidden_layer_sizes=(6, 3, 1),
    max_iter=500,
    activation="relu",
    alpha=0.0005,
    divergence_weight=0,
    batch_size=100000 / 10,
)

# %% {"slideshow": {"slide_type": "skip"}}
vae.fit(x=data)

# %% {"slideshow": {"slide_type": "slide"}}
# %%output filename='../media/04-generative-samples' fig='png'
(
    (
        pd.DataFrame(data, columns=["x", "y"])
        .sample(1000)
        .hvplot.scatter(x="x", y="y", label="Data")
    )
    * (
        pd.DataFrame(vae.sample(1000), columns=["x", "y"]).hvplot.scatter(
            x="x", y="y", label="Samples"
        )
    )
).opts(title="Relu Model Samples from VAE", tools=[])

# %%
kpca = KernelPCA(1, kernel="rbf")
kpca.fit(X=data)

# %% {"slideshow": {"slide_type": "skip"}}
# %%output filename='../media/04-generative-pca' fig='png'
(
    (
        pd.DataFrame(data, columns=["x", "y"])
        .sample(1000)
        .hvplot.scatter(x="x", y="y", label="Data")
    )
    * (
        pd.DataFrame(
            pca.inverse_transform(np.random.normal(size=(1000,)).reshape(-1, 1)),
            columns=["x", "y"],
        ).hvplot.scatter(x="x", y="y", label="Samples")
    )
).opts(title="Relu Model Samples from PCA", tools=[])


# %%
pca = PCA(1)
pca.fit(X=data)

# %%
# %%output filename='../media/04-generative-pca' fig='png'
(
    (
        pd.DataFrame(data, columns=["x", "y"])
        .sample(1000)
        .hvplot.scatter(x="x", y="y", label="Data")
    )
    * (
        pd.DataFrame(
            pca.inverse_transform(np.random.normal(size=(1000,)).reshape(-1, 1)),
            columns=["x", "y"],
        ).hvplot.scatter(x="x", y="y", label="Samples")
    )
).opts(title="Relu Model Samples from PCA", tools=[])
