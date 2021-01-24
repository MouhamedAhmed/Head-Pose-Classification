import os
import shutil
import random
from PIL import Image, ImageOps

paths = [fn for fn in os.listdir('./results/stylegan_poses')
              if (fn.endswith('.jpg'))]

# get frontal paths
frontals = ['004','005']
frontal_paths = []
for path in paths:
    if(path.split('.')[0].split('_')[1] in frontals):
        frontal_paths.append(path)

# make directory for frontal dataset
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'frontal_dataset')
if os.path.exists(final_directory):
    shutil.rmtree(final_directory)
os.makedirs(final_directory)

# save 004 and 005 paths (frontal paths)
for path in frontal_paths:
    src_path = './results/stylegan_poses/'+path
    dst_path = './frontal_dataset/'+path
    shutil.copy2(src_path, dst_path)

# save mirrored version from 004 pose
for path in frontal_paths:
    if(path.split('.')[0].split('_')[1] == '004'):
        im = Image.open('./results/stylegan_poses/'+path)
        im_mirror = ImageOps.mirror(im)
        new_path = path.split('.')[0] + '_mirror.jpg'
        im_mirror.save('./frontal_dataset/'+new_path)

# delete stylegan_poses directory
shutil.rmtree('./results/stylegan_poses')
