"""
https://www.projectpro.io/article/deep-learning-for-image-classification-in-python-with-cnn/418#toc-1
(https://towardsdatascience.com/medical-x-ray-%EF%B8%8F-image-classification-using-convolutional-neural-network-9a6d33b1c2a)
https://keras.io/examples/vision/image_classification_from_scratch/
"""

import tensorflow as tf
import keras
from keras import layers, utils, callbacks, optimizers, preprocessing
from pathlib import Path
import pandas
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE_PATH = "/home/felix/Documents/University/SS2022/ML4B/Data Set Chest X-Ray"
IMAGE_SIZE = (256, 256)
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
    """ loads images as pandas DataFrame for either training, testing or validation """

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


def create_model(input_shape, num_classes):
    """ create model """

    # define parameters and augment data
    inputs = keras.Input(shape=input_shape)
    data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1)])
    x = data_augmentation(inputs)

    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def train_model():
    model = create_model(input_shape=IMAGE_SIZE + (3,), num_classes=2)
    # keras.utils.plot_model(model, show_shapes=True)
    train_ds = create_dataset("train")
    val_ds = create_dataset("val")

    epochs = 50

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """ https://keras.io/examples/vision/grad_cam/ """

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """ https://keras.io/examples/vision/grad_cam/ """

    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)


# grad cam
img_path = "cam.jpg"

model = keras.models.load_model("save_at_50.h5")
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
LAST = "global_average_pooling2d"
img = keras.utils.load_img(
    "/home/felix/Documents/University/SS2022/ML4B/Data Set Chest X-Ray/test/PNEUMONIA/person3_virus_17.jpeg",
    target_size=IMAGE_SIZE)
img_array = keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
i = make_gradcam_heatmap(img_path, make_gradcam_heatmap(img_array, model, LAST))
plt.imshow(i.astype("uint8"))
plt.show()

# rest
"""val_ds = create_dataset("val")
model = keras.models.load_model("save_at_50.h5")
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

idg = preprocessing.image.ImageDataGenerator()
test = idg.flow_from_directory(BASE_PATH + "/test", class_mode="binary", batch_size=BATCH_SIZE)

ds = keras.utils.image_dataset_from_directory(
    BASE_PATH + "/test",
    validation_split=0.2,
    subset="testing",
    seed=1337,
    image_size=IMAGE_SIZE,  # resize images
    batch_size=BATCH_SIZE  # define batch size
)

print(model.evaluate(ds, verbose=0))
results = []
for images in os.walk(BASE_PATH + "/test/PNEUMONIA"):
    for image in images[2]:
        img_path = "/home/felix/Documents/University/SS2022/ML4B/Data Set Chest X-Ray/test/PNEUMONIA/" + image
        img = keras.utils.load_img(
            img_path, target_size=IMAGE_SIZE
        )
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(x=img_array)
        score = predictions[0]
        res = 100 * score
        results.append(res)
print(sum(results) / len(results))
# 38.38% on normal
# 99.63 on pneumonia

print(
    "This image is %.2f percent normal and %.2f percent pneumonia."
    % (100 * (1 - score), 100 * score)
)"""
