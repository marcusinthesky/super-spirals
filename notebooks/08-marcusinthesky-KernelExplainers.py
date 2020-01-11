#%%
import holoviews as hv
import pandas as pd
import hvplot.pandas
from sklearn.datasets import load_boston
from super_spirals.inspection import LimeTabularExplainer
hv.extension('bokeh')

#%%
data = load_boston()
X = data.data
y = data.target

#%%
from sklearn.neural_network import MLPRegressor
lm = MLPRegressor()
lm.fit(X, y)

#%%
lime_explainer = LimeTabularExplainer(n_iter=100)
lime_explainations = lime_explainer.fit_transform(X, y)

# %%
(pd.DataFrame(lime_explainer.feature_importances(pd.np.array([0])).numpy(), columns=data.feature_names)
 .hvplot.bar(title='LIME Feature Importances')
 .opts(xrotation=45))