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
# # ! kaggle datasets download -d kinguistics/heartbeat-sounds -p ../data/raw

# %%
import os
import glob
import zipfile
import sys
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
from toolz.curried import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold
from scipy.io import wavfile  # get the api
from scipy.fftpack import fft
from scipy.spatial import procrustes
from scipy.stats import iqr
import panel as pn
import param
from random import sample


sys.path.append("../")
hv.extension("bokeh")

# %%
from super_spirals.neural_network import VAE

# %%
data_path = os.path.join("..", "data", "raw", "heatbeat-sounds")

if not os.path.exists(data_path):
    with zipfile.ZipFile(
        os.path.join("..", "data", "raw", "heartbeat-sounds.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall(data_path)

# %%
heartbeat_path = os.path.join(data_path, "set_b")

# %%
files = pipe(
    os.listdir(heartbeat_path),
    map(str),
    map(lambda f: os.path.join(heartbeat_path, f)),
    list,
)

# %%
set_a = pipe(data_path,
             lambda f: os.path.join(f, 'set_a.csv'),
             pd.read_csv)

# %%
set_b = pipe(data_path,
             lambda f: os.path.join(f, 'set_b.csv'),
             pd.read_csv)

# %%
from scipy.integrate import simps


# %%
def get_waveform(f, components=10):
    try:
        return (pipe(f,
                     wavfile.read, 
                     get(1),
                     lambda x: x[:20000],
                     lambda x: (x)/(np.quantile(np.abs(x), 0.9)),
                     partial(fft, n=components),
                     np.real,
                     pd.Series))
                        
    except:
        return pd.Series(np.zeros(components) * np.nan)


# %%
frequencies_a = (set_a
                 .fname
                 .apply(lambda f: os.path.join(data_path, f))
                 .apply(get_waveform))

# %%
frequencies_b = pipe(files, pd.Series).apply(get_waveform)

# %%
set_a_filtered = set_a.loc[~frequencies_a.isna().all(axis=1),:]

# %%
pipeline = make_pipeline(StandardScaler(),
                         PCA(whiten=True),
                         VAE(hidden_layer_sizes=(5, 3, 2), 
                             max_iter = 500,
                             activation='tanh'))

# %%
pipeline.fit(frequencies_b.dropna())

# %%
pipeline.named_steps['vae'].encoder.summary()

# %%
latent_a = pipeline.transform(frequencies_a.loc[~frequencies_a.isna().all(axis=1),:])

# %%
latent_a_df = (pd.DataFrame(latent_a, columns = ['Component 1', 'Component 2']).add(np.random.uniform(0,0.1, size=(124,2)))
               .assign(label = set_a_filtered.label.fillna('None')))

# %%
clips = set_a.loc[~frequencies_a.isna().all(axis=1),'fname'].to_list()


# %%
class Dashboard(param.Parameterized):
    files = pn.widgets.Select(name="Audio Clip", value=clips[0], options=clips)
    
    @pn.depends("files.value")
    def update(self, index):
        if index:
            self.files.value = clips[index[0]]
        wav_file = pipe(self.files.value,
                        lambda f: os.path.join(data_path, f))

        data = pipe(wav_file, 
                    wavfile.read, 
                    get(1))

        time = pipe(data, lambda x: x[::400]/np.max(np.abs(x)), hv.Curve).opts(
            width=400, xlabel="time", ylabel="waveform", height=300
        )

        frequency = pipe(data, partial(fft, n=1000), np.real, lambda x: x/np.max(np.abs(x)),hv.Curve).opts(
            xlabel="frequency", ylabel="aplitude", width=400, height=300
        )

        return time + frequency

    @pn.depends("files.value")
    def view(self):
        
        latent = latent_a_df.hvplot.scatter(x='Component 1', 
                                           y='Component 2',
                                           color='label',
                                           title='Latent Space of Heartbeat FFT',
                                           width=800,
                                           height=300,
                                           tools=['tap'])
        
        stream = hv.streams.Selection1D(source=latent)
        
        reg = hv.DynamicMap(self.update, kdims=[], streams=[stream])
        
        audio = pn.widgets.Audio(name="Audio", value=pipe(self.files.value,
                                                          lambda f: os.path.join(data_path, f)))

        
        
        return pn.Column(latent, reg, audio)

# %%
d = Dashboard()

# %%
pn.Column(d.files, d.view)

# %%
