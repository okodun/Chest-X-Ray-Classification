"""
https://www.projectpro.io/article/deep-learning-for-image-classification-in-python-with-cnn/418#toc-1
(https://towardsdatascience.com/medical-x-ray-%EF%B8%8F-image-classification-using-convolutional-neural-network-9a6d33b1c2a)
https://keras.io/examples/vision/image_classification_from_scratch/
"""

import tensorflow as tf
import keras
from keras import layers, utils
from pathlib import Path
import pandas
import matplotlib.pyplot as plt
import cv2

BASE_PATH = "/home/felix/Documents/University/SS2022/ML4B/Data Set Chest X-Ray"
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 16


def __check_cuda_gpu() -> bool:
    """ checks whether GPU is detected """

    return tf.test.is_gpu_available(cuda_only=True)


def __plot(data: [], labels: []):
    """ plots given data with labels """

    df = pandas.DataFrame(data)
    df.columns = ['images']
    df['labels'] = labels
    df = df.sample(frac=1).reset_index(drop=True)
    plt.bar(df['labels'].value_counts().index, df['labels'].value_counts().values)
    plt.show()


def __show_sample_images(data: [], labels: []):
    """ shows first 10 images in data (data contains only the paths to these images) """

    plt.figure(figsize=(10, 5))
    for i in range(0, 10):
        plt.subplot(2, 5, i + 1)
        img = cv2.imread(str(data[i]))
        img = cv2.resize(img, IMAGE_SIZE)
        plt.imshow(img)
        plt.title(labels[i])
        plt.axis("off")
    plt.show()


def get_data(base_path: str, mode: str) -> ():

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
    normal_images = normal_dir.glob("*.jpeg")
    for n in normal_images:
        train_data.append(n)
        train_labels.append("NORMAL")
    # unhealthy cases
    pneumonia_dir = wd / "PNEUMONIA"
    pneumonia_images = pneumonia_dir.glob("*.jpeg")
    for p in pneumonia_images:
        train_data.append(p)
        train_labels.append("PNEUMONIA")

    return train_data, train_labels


def create_dataset(mode: str):
    """ creates a dataset that can be used in an ML model """

    #  raise exception if mode is unknown
    if mode not in ("test", "train", "val"):
        raise ValueError

    subsets = {
        "test": "testing",
        "train": "training",
        "val": "validation"
    }
    subset_name = subsets.get(mode)

    # create path to dataset
    path = BASE_PATH + "/" + mode

    # create data set
    return keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset=subset_name,
        seed=1337,
        image_size=IMAGE_SIZE,  # resize images
        batch_size=BATCH_SIZE  # define batch size
    )


""" create model """

# data augmentation
data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1)])

train_ds = create_dataset("train")
# val_ds = create_dataset("val")
# test_ds = create_dataset("test")


plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

plt.show()
# save model!!!
