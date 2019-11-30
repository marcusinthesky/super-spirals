#%%
from super_spirals.manifold import ParametricTSNE
import pandas as pd
import hvplot.pandas
import holoviews as hv
from sklearn.datasets import make_s_curve


hv.extension('bokeh')

#%%
X, y = make_s_curve(10000, 0.01)

model = ParametricTSNE(hidden_layer_sizes=(5,2))

#%%
model.fit(X)