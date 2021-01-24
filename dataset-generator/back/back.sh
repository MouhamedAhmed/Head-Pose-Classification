# https://drive.google.com/file/d/11566n6dhJtNl39ImejR31kOSDNw7nAIk/view?usp=sharing

#!/bin/bash
FILE="./simple_images.zip"
if test ! -f $FILE; then
    fileid="11566n6dhJtNl39ImejR31kOSDNw7nAIk"
    filename="simple_images.zip"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
fi
unzip -qq simple_images.zip

# collect all back images in one folder
python collect.py

# augment back images
python data_augmentation.py -src ./back_dataset -dst ./back_dataset_aug

# remove unneeded folders
rm -r back_dataset simple_images

# rename back_dataset_aug to back_dataset
mv back_dataset_aug back_dataset 


