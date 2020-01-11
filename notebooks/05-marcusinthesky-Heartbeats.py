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

# %% {"language": "html"}
# <style>
# div.input {
#     display:none;
# }
# </style>

# %% {"language": "html"}
# <style>
# div.input {
#     display:contents;
# }
# </style>

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # Application

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# __Data__
# ![](../media/05-heartbeat-kaggle.png)

# %% {"slideshow": {"slide_type": "skip"}}
# # ! kaggle datasets download -d kinguistics/heartbeat-sounds -p ../data/raw

# %% {"slideshow": {"slide_type": "skip"}}
import os
import glob
import zipfile
import sys
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
from toolz.curried import pipe, map, compose_left, partial
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold
from scipy.io import wavfile  # get the api
from scipy.fftpack import fft, irfft
from scipy.signal import find_peaks, resample_poly
from scipy.spatial import procrustes
from scipy.stats import iqr
import panel as pn
import param
from random import sample


sys.path.append("../")
hv.extension("bokeh")

# %% {"slideshow": {"slide_type": "skip"}}
from super_spirals.neural_network import VAE

# %% [markdown] {"slideshow": {"slide_type": "skip"}}
# ### Load Data

# %% {"slideshow": {"slide_type": "skip"}}
data_path = os.path.join("..", "data", "raw", "heatbeat-sounds")

if not os.path.exists(data_path):
    with zipfile.ZipFile(
        os.path.join("..", "data", "raw", "heartbeat-sounds.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall(data_path)

# %% {"slideshow": {"slide_type": "skip"}}
heartbeat_path = os.path.join(data_path, "set_b")

# %% {"slideshow": {"slide_type": "skip"}}
files = pipe(
    os.listdir(heartbeat_path),
    map(str),
    map(lambda f: os.path.join(heartbeat_path, f)),
    list,
)

# %% {"slideshow": {"slide_type": "skip"}}
set_a = pipe(data_path, lambda f: os.path.join(f, "set_a.csv"), pd.read_csv)

# %% {"slideshow": {"slide_type": "skip"}}
set_b = pipe(data_path, lambda f: os.path.join(f, "set_b.csv"), pd.read_csv)

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# __Preprocessing__
# Data $\rightarrow$ Standardize $\rightarrow$ Clip $\rightarrow$ Split into single beats $\rightarrow$ Resample resolution $\rightarrow$ Fourier Transform

# %% [markdown] {"slideshow": {"slide_type": "skip"}}
# High pass filter

# %% {"slideshow": {"slide_type": "skip"}}
filter_signal = lambda a: (
    pipe(
        a,
        partial(find_peaks, distance=1000),
        get(0),
        lambda x: pipe(a[x], np.median),
        lambda x: np.clip(a, -x, x),
    )
)

# %% {"slideshow": {"slide_type": "skip"}}
get_signal = lambda f: pipe(f, wavfile.read, get(1))

# %% {"slideshow": {"slide_type": "slide"}}
a = pipe(files[0], get_signal)

b = pipe(files[0], get_signal, filter_signal)

c = pipe(
    b,
    partial(find_peaks, distance=1000),
    get(0),
    lambda x: np.vstack((np.arange(b.shape[0]), b)).T[x, :],
)

(
    hv.Curve(a) * hv.Curve(b).opts(color="orange") * hv.Scatter(c).opts(color="green")
).opts(width=600)


# %% [markdown] {"slideshow": {"slide_type": "skip"}}
# split into individual heartbeats, resample to equal length and get Fourier Transform

# %% {"slideshow": {"slide_type": "skip"}}
def explode(b, components=10):
    return pipe(
        b,
        partial(find_peaks, distance=1000),
        get(0),
        sliding_window(2),
        map(lambda x: b[x[0] : x[1]]),
        map(lambda x: (x) / (np.quantile(np.abs(x), 0.9))),
        map(
            lambda x: x[
                np.round(np.linspace(0, x.shape[0] - 1, num=1000)).astype(np.int)
            ]
        ),
        map(partial(fft, n=components)),
        map(lambda x: np.hstack((np.real(x))).reshape(-1)),
        list,
    )


# %% {"slideshow": {"slide_type": "skip"}}
def get_explotion(f, components=25):
    try:
        return pipe(f, wavfile.read, get(1), partial(explode, components=components))

    except:
        return [np.zeros(int(round(components))) * np.nan]


# %% {"slideshow": {"slide_type": "skip"}}
frequencies_a = set_a.fname.apply(lambda f: os.path.join(data_path, f)).apply(
    get_explotion
)

# %% {"slideshow": {"slide_type": "skip"}}
frequencies_b = pipe(files, pd.Series).apply(get_explotion)

# %% [markdown] {"slideshow": {"slide_type": "skip"}}
# Merge data with frequencies

# %% {"slideshow": {"slide_type": "skip"}}
set_a_freq = (
    set_a.assign(frequencies=frequencies_a)
    .explode("frequencies")
    .reset_index(drop=True)
    .assign(label=lambda d: d.label.fillna("None"))
    .where(lambda d: ~d.label.str.startswith("None"))
    .dropna(how="all")
)

# %% {"slideshow": {"slide_type": "skip"}}
X_a = set_a_freq.frequencies.apply(pd.Series)

# %% {"slideshow": {"slide_type": "skip"}}
set_a_filtered = set_a_freq.loc[~X_a.isna().all(axis=1), :]

# %% {"slideshow": {"slide_type": "skip"}}
X_b = (
    pd.DataFrame({"frequencies": frequencies_b})
    .explode("frequencies")
    .reset_index(drop=True)
    .frequencies.apply(pd.Series)
)

# %% [markdown] {"slideshow": {"slide_type": "skip"}}
# ### Train

# %% {"slideshow": {"slide_type": "skip"}}
pipeline = make_pipeline(
    StandardScaler(),
    PCA(whiten=True),
    VAE(
        hidden_layer_sizes=(15, 10, 2),
        n_iter=1000,
        elbo_weight=10,
        activation="tanh",
    ),
)

# %% {"slideshow": {"slide_type": "skip"}}
pipeline.fit(X_b.dropna())

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# __Model__

# %% {"slideshow": {"slide_type": "-"}}
pipeline.named_steps["vae"].encoder.summary()

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Dashboard

# %% {"slideshow": {"slide_type": "skip"}}
latent_a = pipeline.transform(X_a.dropna())  # .loc[~X_a.isna().all(axis=1),:])

# %% {"slideshow": {"slide_type": "skip"}}
latent_a_df = (
    pd.DataFrame(latent_a, columns=["Component 1", "Component 2"])
    .assign(label=set_a_filtered.label.fillna("None"))
    .reset_index()
    .groupby("label")
    .apply(lambda d: d.sample(250, replace=False))
    .reset_index(drop=True)
    .set_index("index")
)

# %% {"slideshow": {"slide_type": "skip"}}
latent_a_df.head()

# %% {"slideshow": {"slide_type": "skip"}}
print(
    latent_a_df.loc[:, ["Component 1", "label"]]
    .groupby("label")
    .describe()
    .iloc[:, 1:3]
    .to_latex()
)

# %% {"slideshow": {"slide_type": "skip"}}
print(
    latent_a_df.loc[:, ["Component 2", "label"]]
    .groupby("label")
    .describe()
    .iloc[:, 1:3]
    .to_latex()
)

# %% {"slideshow": {"slide_type": "skip"}}
clips = set_a_filtered.loc[latent_a_df.index, "fname"].to_list()


# %% {"slideshow": {"slide_type": "skip"}}
class Dashboard(param.Parameterized):
    files = pn.widgets.Select(name="Audio Clip", value=clips[0], options=clips)

    @pn.depends("files.value")
    def update(self, index):
        if index:
            self.files.value = clips[index[0]]
        wav_file = pipe(self.files.value, lambda f: os.path.join(data_path, f))

        data = pipe(wav_file, wavfile.read, get(1))

        time = pipe(data, lambda x: x[::400] / np.max(np.abs(x)), hv.Curve).opts(
            width=400, xlabel="time", ylabel="waveform", height=300
        )

        frequency = pipe(
            data,
            partial(fft, n=1000),
            np.real,
            lambda x: x / np.max(np.abs(x)),
            hv.Curve,
        ).opts(xlabel="frequency", ylabel="aplitude", width=400, height=300)

        return time + frequency

    @pn.depends("files.value")
    def view(self):

        latent = latent_a_df.hvplot.scatter(
            x="Component 1",
            y="Component 2",
            color="label",
            title="Latent Space of Heartbeat FFT",
            width=800,
            size=10,
            height=300,
            tools=["tap"],
        )

        stream = hv.streams.Selection1D(source=latent)

        reg = hv.DynamicMap(self.update, kdims=[], streams=[stream])

        audio = pn.widgets.Audio(
            name="Audio",
            value=pipe(self.files.value, lambda f: os.path.join(data_path, f)),
        )

        return pn.Column(latent, reg, audio)


# %% {"slideshow": {"slide_type": "skip"}}
d = Dashboard()

# %% {"slideshow": {"slide_type": "-"}}
pn.Column(d.files, d.view)

# %% {"slideshow": {"slide_type": "skip"}}
