#!/bin/sh
# generate dirctory for dataset
if [ -d "../dataset" ] 
then
    rm -r ../dataset
fi
mkdir ../dataset

# generate back dataset using google search image API followed by data augmentation
echo back dataset generation
cd back
bash ./back.sh
cd ..
mv ./back/back_dataset ../dataset

# generate frontal dataset with different orientations using InterfaceGAN to navigate on POSE AXIS in StyleGAN Latent Space
echo front dataset generation
cd front
bash ./front.sh 1000
cd ..
mv ./front/frontal_dataset ../dataset

# generate side dataset using persons headpose dataset followed by data augmentation
echo side dataset generation
cd side
bash ./side.sh
cd ..
mv ./side/side_dataset ../dataset



