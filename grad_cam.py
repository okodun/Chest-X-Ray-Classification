"""
please note: this class is almost 100% from https://keras.io/examples/vision/grad_cam/ (we implemented minor changes)
"""

import tensorflow as tf
import keras
from keras import utils
import matplotlib.cm as cm
import numpy as np
import cv2


class GradCam(object):
    """ class for GradCam implementation """

    def __init__(self, path_to_model: str):
        """ initializes GradCam and determines important variables """

        # load model
        self.MODEL = keras.models.load_model(path_to_model)

        # find last convolutional layer
        for idx in range(len(self.MODEL.layers) - 1, 0, -1):
            if "conv" in self.MODEL.layers[idx].name:
                self.LAST_CONVOLUTIONAL_LAYER = self.MODEL.layers[idx].name
                break
            elif idx == 0:
                raise ValueError

        # determine image size
        config = self.MODEL.get_config()["layers"][0]["config"]["batch_input_shape"]
        self.IMAGE_SIZE = config[1], config[2]

    def create(self, path_to_image: str, pred_index=None, cam_path="grad_cam.jpg", alpha=0.4):
        """ creates a heatmap and determines the decision criteria of a trained model """

        # create model with that maps input to output
        model = self.MODEL
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(self.LAST_CONVOLUTIONAL_LAYER).output, model.output]
        )

        # load image as array
        # img_array = GradCam.__load_image_array(path_to_image, self.IMAGE_SIZE)
        image = keras.utils.load_img(path_to_image, target_size=self.IMAGE_SIZE)
        image_array = keras.utils.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(image_array)
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
        heatmap = heatmap.numpy()

        # Load the original image
        img = keras.utils.load_img(path_to_image)
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

    def annotate(self, path_to_image: str):
        img = cv2.imread(path_to_image)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_range = np.array([0, 0, 0], np.uint8)
        upper_range = np.array([26, 255, 255], np.uint8)

        mask = cv2.inRange(hsv, lower_range, upper_range)
        xy = cv2.findNonZero(mask)
        if xy.size > 0:
            i = int(len(xy) / 2)
            # image = cv2.imread("result.jpeg")
            # cv2.line(image, (xy[i][0][0], xy[i][0][1]), (xy[i][0][0] + 400, xy[i][0][1] + 100), (0, 0, 0), 10)
            cv2.putText(img, "important", (xy[i][0][0] - 50, xy[i][0][1]), 1, 1, (0, 0, 0), 2, cv2.LINE_4)
            cv2.imwrite(path_to_image, img)
