import os
import random

PATH = "/home/felix/Documents/University/SS2022/ML4B/Data Set Chest X-Ray/chest_xray/train"


def find_image(mode: str) -> str:
    """ returns a random image """

    # determine path
    if mode == "normal":
        path = PATH + "/NORMAL"
    elif mode == "pneumonia":
        path = PATH + "/PNEUMONIA"
    else:
        raise ValueError

    # get images
    images = next(os.walk(path))[2]

    # find and return image
    return images[random.randint(0, len(images) - 1)]


if __name__ == '__main__':
    print(find_image("pneumonia"))
