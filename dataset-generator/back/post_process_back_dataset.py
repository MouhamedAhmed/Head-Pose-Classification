import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
from PIL import ImageOps

def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    im = Image.fromarray(im)
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    new_im_flipped = ImageOps.mirror(new_im)
    return np.array(new_im), np.array(new_im_flipped)

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'back_dataset_cropped')
if not os.path.exists(final_directory):
   os.makedirs(final_directory)
else:
    shutil.rmtree(final_directory)
    os.makedirs(final_directory)

files = [f for f in os.listdir('./back_dataset')]

for i in range(len(files)):
    #Loading the image to be tested
    image = cv2.imread('./back_dataset/'+files[i])

    image = image[0:int(0.85*len(image)),:,:]

    image, flipped_image = make_square(image)

    cv2.imwrite('./back_dataset_cropped/'+files[i],image)
    cv2.imwrite('./back_dataset_cropped/f_'+files[i],flipped_image)
