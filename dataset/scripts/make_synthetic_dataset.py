"""This script create a synthetic classification dataset to train pipelines onto."""
import os

import numpy as np
from sklearn.datasets import make_classification

# TODO: wrap with argparse
if __name__ == "__main__":
    features, target = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=14,
        n_redundant=2,
        n_repeated=1,
        n_classes=10,
        n_clusters_per_class=2,
        flip_y=0.05,
        shuffle=False,
        random_state=1,
    )

    with open(os.path.join("../synthetic_dataset", "features.npy"), "wb") as f:  # TODO: get path from this file name
        np.save(f, features)
    with open(os.path.join("../synthetic_dataset", "target.npy"), "wb") as f:  # TODO: get path from this file name
        np.save(f, target)
