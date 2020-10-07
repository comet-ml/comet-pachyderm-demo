"""Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
"""
from __future__ import print_function

import os
import logging
import sys

from os import walk
from os.path import join
from urllib.parse import urlparse, urlunparse, urljoin
from urllib.request import getproxies

# fmt: off
from comet_ml import ExistingExperiment, Experiment
from comet_ml.offline import OfflineExperiment

import numpy as np
import keras
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop

# fmt: on


def get_mnist_data(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]
    return (x_train, y_train), (x_test, y_test)


def main():

    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = get_mnist_data(
        path="/pfs/mnist-input/mnist.npz"
    )

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    train(x_train, y_train, x_test, y_test)


def build_model_graph(input_shape=(784,)):
    model = Sequential()
    model.add(Dense(128, activation="sigmoid", input_shape=(784,)))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
    )

    return model


def url_join(base, *parts):
    """ Given a base and url parts (for example [workspace, project, id]) returns a full URL
    """
    # TODO: Enforce base to have a scheme and netloc?
    result = base

    for part in parts[:-1]:
        if not part.endswith("/"):
            raise ValueError("Intermediary part not ending with /")

        result = urljoin(result, part)

    result = urljoin(result, parts[-1])

    return result


def list_files(path):
    return os.listdir(path)


def get_pachyderm_meta():
    PACH_BASE_URL = os.environ["PACH_BASE_URL"]
    INPUT_REPO_NAME = os.environ["INPUT_REPO_NAME"]
    INPUT_FILES = list_files(os.path.join("/pfs/", INPUT_REPO_NAME))
    OUTPUT_FILES = list_files("/pfs/out/")

    pachyderm_meta = {
        "env": {
            "PACH_JOB_ID": os.environ["PACH_JOB_ID"],
            "PACH_NAMESPACE": os.environ["PACH_NAMESPACE"],
            "PACH_OUTPUT_COMMIT_ID": os.environ["PACH_OUTPUT_COMMIT_ID"],
            "PPS_PIPELINE_NAME": os.environ["PPS_PIPELINE_NAME"],
            "PPS_POD_NAME": os.environ["PPS_POD_NAME"],
            "PPS_SPEC_COMMIT": os.environ["PPS_SPEC_COMMIT"],
            "STORAGE_BACKEND": os.environ["STORAGE_BACKEND"],
            "mnist-input": os.environ["mnist-input"],
            "mnist-input_COMMIT": os.environ["mnist-input_COMMIT"],
        },
        "url": {
            "base_url": PACH_BASE_URL,
            "inputs": {
                "mnist-input": {
                    "commit": url_join(
                        "app/repos/",
                        INPUT_REPO_NAME + "/",
                        "commits/",
                        os.environ["mnist-input_COMMIT"],
                    ),
                    "repo": url_join("app/repos/", INPUT_REPO_NAME),
                    "files": {
                        file_name: url_join(
                            "app/repos/",
                            INPUT_REPO_NAME + "/",
                            "commits/",
                            os.environ["mnist-input_COMMIT"] + "/",
                            "files/",
                            file_name,
                        )
                        for file_name in INPUT_FILES
                    },
                }
            },
            "job": url_join("app/jobs/", os.environ["PACH_JOB_ID"]),
            "output_commit": url_join(
                "app/repos/",
                os.environ["PPS_PIPELINE_NAME"] + "/",
                "commits/",
                os.environ["PACH_OUTPUT_COMMIT_ID"],
            ),
            "output_files": {
                file_name: url_join(
                    "app/repos/",
                    os.environ["PPS_PIPELINE_NAME"] + "/",
                    "commits/",
                    os.environ["PACH_OUTPUT_COMMIT_ID"] + "/",
                    "files/",
                    file_name,
                )
                for file_name in OUTPUT_FILES
            },
            "pipeline": url_join("app/pipelines/", os.environ["PPS_PIPELINE_NAME"]),
        },
        "input_files": INPUT_FILES,
        "output_files": OUTPUT_FILES,
    }

    other_data = {
        "pachyderm_job_id": os.environ["PACH_JOB_ID"],
        "pachyderm_pipeline_name": os.environ["PPS_PIPELINE_NAME"],
        "pachyderm_input_commit": os.environ["mnist-input_COMMIT"],
        "pachyderm_output_commit": os.environ["PACH_OUTPUT_COMMIT_ID"],
        "pachyderm_input_repo_name": INPUT_REPO_NAME,
    }

    return pachyderm_meta, other_data


def train(x_train, y_train, x_test, y_test):
    experiment = Experiment()
    experiment.log_dataset_hash(x_train)

    # Define model
    model = build_model_graph()

    model.fit(
        x_train, y_train, batch_size=120, epochs=10, validation_data=(x_test, y_test)
    )
    experiment.send_notification("Training done", "finished", {"Data": "100"})
    score = model.evaluate(x_test, y_test, verbose=0)
    logging.info("Score %s", score)
    model.save("/pfs/out/model.h5")

    meta, other = get_pachyderm_meta()
    experiment.log_asset_data(meta, name="pachyderm-meta.json")
    experiment.log_others(other)


if __name__ == "__main__":
    main()

