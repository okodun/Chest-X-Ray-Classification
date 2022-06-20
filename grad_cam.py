"""
please note: this class is almost 100% from https://keras.io/examples/vision/grad_cam/ (we implemented minor changes)
"""

import numpy as np
import tensorflow as tf
from keras import utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import keras

# constants
IMAGE_SIZE = (256, 256)
LAST_CONV_LAYER_NAME = "conv2d_5"
IMAGE_PATH = "/home/felix/Documents/University/SS2022/ML4B/Data Set Chest X-Ray/test/NORMAL/IM-0011-0001-0002.jpeg"
MODEL_PATH = "save_at_50.h5"


class GradCam(object):
    def __init__(self, img_path: str, model_path: str, last_conv_layer_name, pred_index=None):
        raise NotImplementedError


def show(image_path: str, image_size: ()):
    """ plots and shows image """

    image = keras.utils.load_img(image_path, target_size=image_size)
    image_array = keras.utils.img_to_array(image)
    plt.imshow(image_array.astype("uint8"))
    plt.show()


def __load_image_array(image_path: str, image_size: ()):
    """ loads an image as an array with additional batch """

    image = keras.utils.load_img(image_path, target_size=image_size)
    image_array = keras.utils.img_to_array(image)
    return np.expand_dims(image_array, axis=0)


def create_heatmap(img_path: str, model_path: str, last_conv_layer_name, pred_index=None):
    """ creates a heatmap and determines the decision criteria of a trained model """

    # create model with that maps input to output
    model = keras.models.load_model(model_path)
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # load image as array
    img_array = __load_image_array(img_path)

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
    # pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    pooled_grads = tf.math.reduce_mean(grads, axis=(0, 1))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam2.jpg", alpha=0.4):
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


if __name__ == '__main__':
    # prepare image
    img_array = __load_image_array(IMAGE_PATH)

    # make model
    model = keras.models.load_model("save_at_50.h5")

    # Print what the top predicted class is
    preds = model.predict(x=img_array)
    print("PNEUMONIA: %.2f" % (100 * preds[0]))

    # Remove last layer's activation function
    model.layers[-1].activation = None

    # Generate class activation heatmap
    heatmap = create_heatmap(IMAGE_PATH, MODEL_PATH, LAST_CONV_LAYER_NAME)

    # save heatmap
    save_and_display_gradcam(IMAGE_PATH, heatmap)
