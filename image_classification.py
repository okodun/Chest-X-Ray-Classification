"""
https://www.projectpro.io/article/deep-learning-for-image-classification-in-python-with-cnn/418#toc-1
"""

import tensorflow as tf
from pathlib import Path
import pandas
import matplotlib.pyplot as plt

BASE_PATH = "/home/felix/Documents/University/SS2022/ML4B/Data Set Chest X-Ray"


def __check_cuda_gpu() -> bool:
    """ checks whether GPU is detected """

    return tf.test.is_gpu_available(cuda_only=True)


def load_xrays(base_path: str, mode: str) -> pandas.DataFrame:
    """ loads images as pandas DataFrame for either training, testing or valuation """

    # raise exception if mode is unknown
    if mode not in ("test", "train", "val"):
        raise ValueError

    # find paths to X-Ray images
    home_dir = Path(base_path)
    wd = home_dir / mode

    # load images for training
    train_data = []
    train_labels = []
    # normal cases
    normal_dir = wd / "NORMAL"
    normal = normal_dir.glob("*.jpeg")
    for n in normal:
        train_data.append(n)
        train_labels.append("NORMAL")
    # unhealthy cases
    pneumonia_dir = wd / "PNEUMONIA"
    pneumonia = pneumonia_dir.glob("*.jpeg")
    for p in pneumonia:
        train_data.append(p)
        train_labels.append("PNEUMONIA")

    # plot number of images
    df = pandas.DataFrame(train_data)
    df.columns = ['images']
    df['labels'] = train_labels
    df = df.sample(frac=1).reset_index(drop=True)
    return df


test = load_xrays(BASE_PATH, "val")
plt.bar(test['labels'].value_counts().index, test['labels'].value_counts().values)
plt.show()
