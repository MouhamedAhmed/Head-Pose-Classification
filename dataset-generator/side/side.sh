# https://drive.google.com/file/d/1vRF4ZM6qJrWNPqDazCh1vs3VFCIaWj0b/view?usp=sharing

#!/bin/bash
FILE="./Head-Pose-Annotations-Dataset.zip"
if test ! -f $FILE; then
    fileid="1vRF4ZM6qJrWNPqDazCh1vs3VFCIaWj0b"
    filename="Head-Pose-Annotations-Dataset.zip"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
fi
unzip -qq Head-Pose-Annotations-Dataset.zip

# collect all side images in one folder
python collect.py

# augment side images
python data_augmentation.py -src ./side_dataset -dst ./side_dataset_aug

# remove unneeded folders
rm -r Head-Pose-Annotations-Dataset side_dataset

# rename side_dataset_aug to side_dataset
mv side_dataset_aug side_dataset 
