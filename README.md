# Installation
Create Conda environment with pytorch and tensorflow 1.15 needed for interfaceGAN and install other dependencies on it from `requirements.txt`.
```bash
conda create -n HeadPoseEstimator tensorflow-gpu==1.15 pytorch torchvision
conda activate HeadPoseEstimator
conda install pip
pip3 install -r requirements.txt
conda deactivate
```

# Dataset Preparation
## Frontal faces
Frontal faces are generated using **InterfaceGAN** which navigates in **StyleGAN** axes, we used **POSE** axis to generate different poses that all can be considered as frontal.

## Side faces
Side faces are generated from [Head Pose Annotations Dataset](https://yekara.com/headpose_annotations/) followed by different augmnetation techniques.
PS. We didn't use InterfaceGAN to generate side faces as it generates very poor results.

## Back faces
Back faces are generated from 'Google Image Search' using `simple_image_download` library for these searches
- back hair style
- back head
- back head hair
- back head hair cut
- back head hair girls
- black man head back
- blonde head girl back
- blonde man head back
- head man back
- hijabi girl head back
- muslim girl back
- Red head girl back
<br />
 for the first 100 search results then they are filtered manually then different data augmentation techniques are used...

## Usage
activate our conda environment and run `generate_dataset.sh` in `dataset-generator` folder
```bash
conda activate HeadPoseEstimator
cd dataset-generator
bash generate_dataset.sh
cd ..
conda deactivate
```

# Model Training
We tried both ResNet18 and a simple architecture, both followed by a linear layer. Both architectures performed well on our collected dataset.
We tried contrastive loss along with cross entropy loss and it didn't make a difference.

## Simple Architecture Details
- we used 7 CNN layers and 1 fully-connected layer 
- first 4 CNN layers used kernel size of 5
- last 3 CNN layers used kernel size of 3
- all CNN layers are followed by (BatchNorm-ReLU-MaxPool)

## To Train
### Prepare Dataset
```bash
cd train
python trainer.py --do_data -dataset_path <path_to_dataset>
cd ..
```
Where **path_to_dataset** is path to dataset folder that contains folder for each label

### Train
```bash
cd train
python trainer.py --do_train -arch <model_architecture> -lr <learning_rate> -lr_decay <learning_rate_decay> -lr_decay_step_size <learning_rate_decay_step_size> -batch <batch_size> -epochs <num_of_epochs> -size <img_size>
cd ..
```
Where 
<br />
**model_architecture** is **simple** or **resnet**, default is simple <br />
**learning_rate** is the starting learning rate, default is 0.001 <br />
**learning_rate_decay** is the decay rate of learning rate after each step size, default is 0.2 <br />
**learning_rate_decay_step_size** is the number of epochs after which decay learning rate occurs, default is 5 <br />
**batch_size** is the maximum number of samples per batch, default is 32 <br />
**num_of_epochs** is the total number of epochs, default is 50 <br />
**img_size** is the size to downsample images to before training, default is 256 

### Model Training Log: Convergence after 4 to 5 Epochs
    --- Epoch: 0	Train acc: 0.5781	Valid acc: 0.5389	
    --- Epoch: 1	Train acc: 0.9536	Valid acc: 0.9197	
    --- Epoch: 2	Train acc: 0.9810	Valid acc: 0.9588	
    --- Epoch: 3	Train acc: 0.9937	Valid acc: 0.9805	
    --- Epoch: 4	Train acc: 0.9937	Valid acc: 0.9802	
    --- Epoch: 5	Train acc: 1.0000	Valid acc: 0.9838	
    --- Epoch: 6	Train acc: 0.9979	Valid acc: 0.9814	
    --- Epoch: 7	Train acc: 1.0000	Valid acc: 0.9826

#  Model Predict
Use **Predictor** class in `predict.py`
```python
predictor = Predictor(model_path, labels_path)
predicted_label = predictor.predict(img_path)
```
Where <br />
**model_path** is the path to `model.pkl` file generated from training <br />
**labels_path** is the path to `labels.npy` file generated from do_data <br />
**img_path** is tha path to image needed to be predicted

