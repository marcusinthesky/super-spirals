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
# ! kaggle datasets download -d kinguistics/heartbeat-sounds -p ../data/raw

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
import panel as pn
from random import sample


sys.path.append("../")
hv.extension("bokeh")

# %%
from lib.neural_network import VAE

# %%
unzip_path = os.path.join("..", "data", "raw", "heatbeat-sounds")

if not os.listdir(unzip_path):
    with zipfile.ZipFile(
        os.path.join("..", "data", "raw", "heartbeat-sounds.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall(unzip_path)

# %%
heartbeat_path = os.path.join(unzip_path, "set_a")

# %%
files = pipe(
    os.listdir(heartbeat_path),
    map(str),
    map(lambda f: os.path.join(heartbeat_path, f)),
    list,
)


# %%

# %%
class Dashboard(param.Parameterized):
    files = pn.widgets.Select(name="Audio Clip", value=files[0], options=files)

    @pn.depends("files.value")
    def view(self):
        wav_file = self.files.value

        audio = pn.widgets.Audio(name="Audio", value=wav_file)

        data = pipe(wav_file, wavfile.read, get(1))

        time = pipe(data, lambda x: x[::10], hv.Curve).opts(
            width=400, xlabel="time", ylabel="waveform", height=300
        )

        frequency = pipe(data, partial(fft, n=1000), np.real, hv.Curve).opts(
            xlabel="frequency", ylabel="aplitude", width=400, height=300
        )

        return pn.Column(time + frequency, audio)


# %%
dashboard = Dashboard()

# %%
pn.Column(dashboard.files, dashboard.view)

# %%
