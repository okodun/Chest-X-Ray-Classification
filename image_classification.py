"""
https://www.projectpro.io/article/deep-learning-for-image-classification-in-python-with-cnn/418#toc-1
(https://towardsdatascience.com/medical-x-ray-%EF%B8%8F-image-classification-using-convolutional-neural-network-9a6d33b1c2a)
"""

import tensorflow as tf
import keras
from keras import layers, utils, callbacks, optimizers
from pathlib import Path
import pandas
import cv2
import matplotlib.pyplot as plt

# constants
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
    """
    Creates a keras pipeline.
    Pipeline created by keras.com (https://keras.io/examples/vision/image_classification_from_scratch/)
    """

    # define parameters and augment data
    inputs = keras.Input(shape=input_shape)
    data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1)])
    x = data_augmentation(inputs)

    # scale images and create first convolutional layer
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # second convolutional layer
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Set aside residual
    previous_block_activation = x

    # add convolutional layers for different sizes
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

        # Add back residual
        x = layers.add([x, residual])

        # Set aside next residual
        previous_block_activation = x

    # last convolutional layer
    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # last pooling layer
    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    # output layer
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def train_model(epochs: int, path_to_model: str):
    """  trains a keras model """

    # create model
    model = create_model(input_shape=IMAGE_SIZE + (3,), num_classes=2)

    # load necessary data sets
    train_ds = create_dataset("train")
    val_ds = create_dataset("val")

    # compile model
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # actually train model
    model.fit(train_ds[0], train_ds[1], epochs=epochs, callbacks=callbacks, validation_data=(val_ds[0], val_ds[1]))

    # save model
    model.save(path_to_model)


def predict(path_to_image: str, path_to_model: str):
    """ categorizes a given image """

    # load model
    model = keras.models.load_model("save_at_50.h5")

    """model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )"""

    # load image
    image = keras.utils.load_img(path_to_image, target_size=IMAGE_SIZE)
    image_array = keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)

    # predict
    predictions = model.predict(x=image_array)

    # score for pneumonia -> normal = 1 - score
    return predictions[0]


res = predict(
    "/home/felix/Documents/University/SS2022/ML4B/Data Set Chest X-Ray/test/PNEUMONIA/person85_bacteria_422.jpeg",
    "save_at_50.h5")
print("Pneumonia: %.4f%% | Normal: %.4f%%" % (100 * res, 100 * (1 - res)))

"""# testing
val_ds = create_dataset("val")
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
    subset="training",
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
