import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
from typing import Dict
from toolz.curried import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.base import TransformerMixin


def get_latent_space(
    model, X_train, X_test, y_train, y_test, tag, labels, feature_names
):

    latent = pipe(
        X_test,
        model.transform,
        partial(pd.DataFrame, columns=["Component 1", "Component 2"]),
    )

    return (
        pd.concat([latent], axis=1)
        .assign(label=y_test)
        .assign(label=lambda d: d.label.replace(labels))
        .assign(tag=tag)
    )


def reconstruction_benchmark(
    dataset: Dict[str, np.ndarray], models: Dict[str, TransformerMixin], label: str
):
    """
    """
    data = dataset

    if hasattr(data, "target_names"):
        labels = dict(enumerate(data.target_names))
    else:
        labels = pipe(data.target, np.unique, map(str), list, np.array).astype(str)

    if hasattr(data, "feature_names"):
        names = dict(enumerate(data.feature_names))
    else:
        names = pipe(range(data.data.shape[1]), map(str), list, np.array).astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.33, random_state=42
    )

    for m in models.values():
        m.fit(X_train)

    latent = pd.concat(
        [
            get_latent_space(m, X_train, X_test, y_train, y_test, t, labels, names)
            for t, m in models.items()
        ]
    )

    reconstruction_loss = pd.DataFrame(
        {
            t: pipe(
                X_train,
                m.transform,
                m.inverse_transform,
                lambda x: np.subtract(x, X_train),
                lambda x: np.power(x, 2),
                lambda x: np.array(x).flatten(),
                np.mean,
            )
            for t, m in models.items()
        },
        index=["Reconstruction Loss"],
    ).T

    silhouette = pd.DataFrame(
        {
            t: [silhouette_score(X=pipe(X_test, m.transform), labels=y_test)]
            for t, m in models.items()
        },
        index=["Silhoutte"],
    ).T

    return (
        latent,
        reconstruction_loss.hvplot.bar(title=f"{label}: Reconstruction Loss"),
        silhouette.hvplot.bar(title=f"{label}: Silhouette Scores"),
        latent.hvplot.scatter(
            x="Component 1", y="Component 2", color="label", groupby="tag", label=label
        )
        .layout()
        .cols(2),
    )
