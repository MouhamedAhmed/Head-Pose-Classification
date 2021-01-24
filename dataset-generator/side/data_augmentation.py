import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np 
import albumentations as A
import random
from PIL import Image
import sys
import os
import progressbar
import argparse
def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    im = Image.fromarray(im)
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return np.array(new_im)

def main(src_path_to_dataset, dst_path_to_dataset):
    if not os.path.exists(dst_path_to_dataset):
        os.makedirs(dst_path_to_dataset)
    included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']
    images_paths = [src_path_to_dataset + '/' + fn for fn in os.listdir(src_path_to_dataset)
                if (fn.endswith(ext) for ext in included_extensions)]

    bar = progressbar.ProgressBar(maxval=len(images_paths), \
                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    c = 0

    for img_path in images_paths:
        img_name = (img_path.split('/')[-1]).split('.')[0]
            
        image = np.array(Image.open(img_path))
        try:
            image = image[0:int(0.85*len(image)),:,:]
        except:
            pass

        ###############################################
        # define an augmentation pipeline
        aug_pipeline = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.GaussianBlur((0, 2.0))), # apply Gaussian blur with a sigma between 0 and 3 to 50% of the images
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(5, 15))),
            iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.2))),
            iaa.Sometimes(0.7, iaa.AddToHueAndSaturation((-30, 30))),
            iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=90, sigma=9)),
            iaa.Sometimes(0.5, iaa.Fliplr(1.0)), # horizontally flip    
            iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))), # sharpen images
            iaa.Sometimes(0.5, iaa.Affine(rotate=5)) # rotate 50% of the images
        ],
        random_order=True # apply the augmentations in random order
        )

        # apply augmentation pipeline to sample image
        images_aug = []
        for _ in range(3):
            try:
                images_aug.append(aug_pipeline.augment_image(image))
            except:
                continue
        images_aug = np.array(images_aug)

        bright_contrast = A.RandomBrightnessContrast(p=1) # random brightness and contrast
        gamma = A.RandomGamma(p=1) # random gamma
        clahe = A.CLAHE(p=1) # CLAHE (see https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE)
        blur = A.Blur()

        total_augmented_images = []
        for img in images_aug:
            total_augmented_images.append(img)
            if random.random() < 0.4:
                img_bc = bright_contrast(image = img)
                total_augmented_images.append(img_bc['image'])
            if random.random() < 0.4:
                img_gamma = gamma(image = img)
                total_augmented_images.append(img_gamma['image'])
            if random.random() < 0.4:
                img_clahe = clahe(image = img)
                total_augmented_images.append(img_clahe['image'])
            if random.random() < 0.4:
                img_blur = blur(image = img)
                total_augmented_images.append(img_blur['image'])

        try:
            total_augmented_images = np.array(total_augmented_images)
            total_sq_images = np.array([make_square(img) for img in total_augmented_images])
            # limit to 3000
            total_sq_images = np.array(random.sample(list(total_sq_images), random.choice([2,3])))
            count = 0
            for i in total_sq_images:
                Image.fromarray(i).save(dst_path_to_dataset + '/' + img_name + '_' + str(count) + '.jpeg')
                count += 1
        except:
            pass
        
        bar.update(c + 1)

        c += 1

    bar.finish()

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-src', '--src_path_to_dataset', type=str, help='path to dataset to be augmented')
    argparser.add_argument('-dst', '--dst_path_to_dataset', type=str, help='path to folder to save augmented dataset in')

    args = argparser.parse_args()

    main(args.src_path_to_dataset, args.dst_path_to_dataset)
