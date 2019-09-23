# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import sys
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
from toolz.curried import *


sys.path.append("../")
hv.extension("bokeh")

# %%
from lib.neural_network import VAE

# %%
X = np.random.multivariate_normal(
    mean=np.zeros(2), cov=np.diag(np.ones(2)), size=100000
)

beta = np.random.uniform(-0.1, 0.1, size=(2, 2))
data = pd.DataFrame(X @ beta).where(lambda d: d > 0).dropna(how="any").to_numpy()

# %%
vae = VAE(hidden_layer_sizes=(2, 1), activation="relu")

# %%
vae.fit(x=data)

# %%
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
