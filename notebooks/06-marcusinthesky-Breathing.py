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
# # ! kaggle datasets download -d vbookshelf/respiratory-sound-database -p ../data/raw

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
from toolz.curried import pipe, map, compose_left, partial
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold
from scipy.io import wavfile  # get the api
from scipy.fftpack import fft
from scipy.stats import iqr
import panel as pn
import param
from random import sample


sys.path.append("../")
hv.extension("bokeh")

# %%
from super_spirals.neural_network import VAE
from super_spirals.io import read_wav

# %%
data_path = os.path.join("..", "data", "raw", "respiratory-sound-database")

if not os.path.exists(data_path):
    with zipfile.ZipFile(
        os.path.join("..", "data", "raw", "respiratory-sound-database.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall(data_path)

# %%
database_path = os.path.join(data_path, "Respiratory_Sound_Database")
if not os.path.exists(database_path):
    with zipfile.ZipFile(
        os.path.join(data_path, "Respiratory_Sound_Database.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall(data_path)

# %%
audio_path = os.path.join(database_path, "audio_and_txt_files")

# %%
audio_files = pipe(
    os.listdir(audio_path),
    map(str),
    map(lambda f: os.path.join(audio_path, f)),
    filter(lambda s: s.endswith("wav")),
    list,
)

# %%
patient_diagnosis = pipe(
    database_path,
    lambda f: os.path.join(f, "patient_diagnosis.csv"),
    partial(pd.read_csv, names=["patient", "diagnosis"]),
)

# %%
filter_signal = lambda a: (
    pipe(
        a,
        partial(find_peaks, distance=1000),
        get(0),
        lambda x: pipe(a[x], np.median),
        lambda x: np.where(abs(a) > x, np.sign(a) * x, a),
    )
)


# %%
def get_waveform(f, components=20):
    try:
        return pipe(
            f,
            read_wav,
            get(1),
            lambda x: x[::250],
            lambda x: (x) / (np.quantile(np.abs(x), 0.9)),
            partial(fft, n=components),
            np.real,
            pd.Series,
        )

    except:
        return pd.Series(np.zeros(components) * np.nan)


# %%
audio_frequencies = pd.Series(audio_files).apply(get_waveform)

# %%
pipeline = make_pipeline(
    StandardScaler(),
    PCA(whiten=True),
    VAE(hidden_layer_sizes=(10, 7, 5, 2), n_iter=500, activation="tanh"),
)

# %%
pipeline.fit(audio_frequencies.dropna())

# %%
pipeline.named_steps["vae"].encoder.summary()

# %%
latent = pipeline.transform(audio_frequencies.dropna())

# %%
not_null_index = (
    audio_frequencies.isna().all(1).where(lambda x: ~x).dropna().index.to_list()
)

# %%
latent_df = (
    pd.DataFrame(latent, columns=["Component 1", "Component 2"])
    .assign(
        file=pd.Series(audio_files).loc[not_null_index].dropna().reset_index(drop=True)
    )
    .assign(
        patient=lambda d: d.file.str.split("/")
        .apply(get(-1))
        .str.split("_")
        .apply(get(0))
        .astype(np.int)
    )
    .merge(patient_diagnosis, how="left", on="patient")
    .groupby("diagnosis")
    .apply(lambda d: d.sample(n=100, replace=True))
    .reset_index(drop=True)
)

# %%
clips = latent_df.file.to_list()


# %%
class Dashboard(param.Parameterized):
    files = pn.widgets.Select(name="Audio Clip", value=clips[0], options=clips)

    @pn.depends("files.value")
    def update(self, index):
        if index:
            self.files.value = clips[index[0]]
        wav_file = pipe(self.files.value)

        data = pipe(wav_file, read_wav, get(1))

        time = pipe(
            data, lambda x: x[::10], lambda x: x / np.max(np.abs(x)), hv.Curve
        ).opts(width=400, xlabel="time", ylabel="waveform", height=300)

        frequency = pipe(
            data,
            partial(fft, n=100),
            np.real,
            lambda x: x / np.max(np.abs(x)),
            hv.Curve,
        ).opts(xlabel="frequency", ylabel="aplitude", width=400, height=300)

        return time + frequency

    @pn.depends("files.value")
    def view(self):

        latent = latent_df.hvplot.scatter(
            x="Component 1",
            y="Component 2",
            color="diagnosis",
            title="Latent Space of Heartbeat FFT",
            width=800,
            height=300,
            tools=["tap"],
        )

        stream = hv.streams.Selection1D(source=latent)

        reg = hv.DynamicMap(self.update, kdims=[], streams=[stream])

        audio = pn.widgets.Audio(name="Audio", value=pipe(self.files.value))

        return pn.Column(latent, reg, audio)


# %%
d = Dashboard()

# %%
pn.Column(d.files, d.view)

# %%
