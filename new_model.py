import os
import re
import shutil
import random

# constants
BASE_PATH = "/home/felix/Documents/University/SS2022/ML4B/Data Set Chest X-Ray"

# define path variables
normal_path = BASE_PATH + "/train/NORMAL"
pneumonia_path = BASE_PATH + "/train/PNEUMONIA_OLD"

# get paths to normal images
normal_dict = {}
images = next(os.walk(normal_path))[2]
for i in range(0, len(images)):
    normal_dict.update({i: images[i]})

# get paths to pneumonia images
pneumonia_dict = {}
images = next(os.walk(pneumonia_path))[2]
for i in range(0, len(images)):
    pneumonia_dict.update({i: images[i]})

# determine how many images are required for each category
virus_length = int(len(normal_dict) / 2) + 1
print(virus_length)
bacteria_length = int(len(normal_dict) / 2)
print(bacteria_length)

# chose images per category randomly
virus_images = []
bacteria_images = []
randoms = []
while virus_length > 0 or bacteria_length > 0:
    # get random image and check for duplicates
    random_int = random.randint(0, len(pneumonia_dict) - 1)
    if random_int in randoms:
        continue
    else:
        randoms.append(random_int)
    # check to which group image belongs
    split = re.split("_", (pneumonia_dict[random_int]))
    if (split[1] == "virus" or split[2] == "virus") and virus_length > 0:
        virus_images.append(pneumonia_path + "/" + pneumonia_dict[random_int])
        virus_length -= 1
    elif (split[1] == "bacteria" or split[2] == "bacteria") and bacteria_length > 0:
        bacteria_images.append(pneumonia_path + "/" + pneumonia_dict[random_int])
        bacteria_length -= 1

# combine lists and shuffle result
pneumonia_images = virus_images + bacteria_images
pneumonia_images = random.sample(pneumonia_images, len(pneumonia_images))

# copy files
new_destination = "/home/felix/Documents/University/SS2022/ML4B/Data Set Chest X-Ray/train/PNEUMONIA/"
for image in pneumonia_images:
    split = re.split("/", image)
    shutil.copy(image, new_destination + split[len(split) - 1])
print(len(next(os.walk(new_destination))[2]))
