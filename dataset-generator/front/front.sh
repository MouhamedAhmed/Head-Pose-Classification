#!/bin/sh
# download model 
# https://drive.google.com/file/d/1Wlz8vbSbnE_AclQJglfSAm5MOZVOB67_/view?usp=sharing

FILE="./models/pretrain/karras2019stylegan-ffhq-1024x1024.pkl"
if test ! -f $FILE; then
    fileid="1Wlz8vbSbnE_AclQJglfSAm5MOZVOB67_"
    filename="./models/pretrain/karras2019stylegan-ffhq-1024x1024.pkl"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
fi

LATENT_CODE_NUM=$1
python edit.py \
    -m stylegan_ffhq \
    -b boundaries/stylegan_ffhq_pose_boundary.npy \
    -n "$LATENT_CODE_NUM" \
    -o results/stylegan_poses

python post_process.py
