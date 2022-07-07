import numpy as np
import keras
from keras import Sequential, layers, utils
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_color_gradients(category, cmap_list):
    """ plots color gradients """

    # map gradients
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.80)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        ax.text(-0.01, 0.5, "low", va='center', ha='right', fontsize=10, transform=ax.transAxes)
        ax.text(279, 0.5, "high", va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    # plot
    plt.show()


# plot_color_gradients('Miscellaneous', ['jet'])

def plot_data(path_to_directory: str) -> pd.DataFrame:
    normal_directory = Path(path_to_directory + "/NORMAL")
    normal_directory = Path(path_to_directory + "/PNEUMONIA")
    normal = normal_directory.glob("*.jpeg")
    pneumonia = normal_directory.glob("*.jpeg")
    data = []
    label = []
    for img in normal:
        data.append(img)
        label.append("NORMAL")
    for img in pneumonia:
        data.append(img)
        label.append("PNEUMONIA")
    df = pd.DataFrame(data)
    df.columns = ["images"]
    df["labels"] = label
    df = df.sample(frac=1).reset_index(drop=True)
    return df


path = "/home/felix/Documents/University/SS2022/ML4B/Data Set Chest X-Ray/train"
"""dframe = plot_data(path)
print(dframe.shape)
plt.bar(dframe["labels"].value_counts().index, dframe["labels"].value_counts().values)"""
dframe = keras.utils.image_dataset_from_directory(path, seed=1337, image_size=(256, 256), batch_size=16)
data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1)])
plt.figure(figsize=(10, 10))
for images, _ in dframe.take(1):
    for i in range(2):
        if i == 0:
            augmented_images = images
        else:
            augmented_images = data_augmentation(images)
        ax = plt.subplot(1, 2, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
plt.show()
