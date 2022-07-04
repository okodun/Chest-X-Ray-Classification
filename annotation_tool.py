"""
this module provides the image annotation tool
"""
import matplotlib.pyplot as plt
import cv2


def annotate(path_to_image: str, save=False):
    """ opens and annotates an image """

    # open image
    image = cv2.imread(path_to_image)

    # prepare plotting
    fig, sub = plt.subplots()
    fig.set_size_inches(9.5, 10)
    sub.imshow(image)
    plt.axis("off")

    # annotate
    sub.annotate('Most important for decision', xy=(800, 1500),
                 bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5),
                 color="white", xytext=(1100, 1515), arrowprops=dict(color='#3eadae', arrowstyle='-', lw='3.5'),
                 xycoords="data")
    sub.annotate('Peak Value', xy=(600, 1100), bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5),
                 color="white", xytext=(0, 1115), arrowprops=dict(color='#3eadae', arrowstyle='-', lw='3.5'),
                 xycoords="data")

    # save if flag is set
    if save:
        plt.savefig("annotated_img.png", transparent=True)
    else:
        plt.show()
