#%%
from super_spirals.manifold import ParametricTSNE
import pandas as pd
import hvplot.pandas
import holoviews as hv
from sklearn.datasets import make_s_curve


hv.extension('bokeh')

#%%
X, y = make_s_curve(10000, 0.01)

model = ParametricTSNE(50., hidden_layer_sizes=(5,2), n_iter=5)

#%%
Z = model.fit_transform(X)

#%%
(pd.DataFrame(Z, columns=['x','y'])
.assign(color=y)
.sample(500)
.hvplot.scatter(x='x',y='y',color='color'))


# %%
