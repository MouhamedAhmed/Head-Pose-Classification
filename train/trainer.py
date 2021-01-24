import argparse
import csv
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from model import Classifier, ResClassifier
from train_val import training_loop
from dataset import PoseDataset
from torch.utils.data import DataLoader

def generate_csv(dataset_path):
    filePath = './labels.csv'
    # remove the csv file if exists
    if os.path.exists(filePath):
        os.remove(filePath)

    labels_folders = os.listdir(dataset_path)
    label = 0
    for folder in labels_folders:
        images = os.listdir(os.path.join(dataset_path, folder))
        for image in images:
            path = os.path.join(dataset_path, folder, image)
            
            with open(filePath, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([path, label])
        label += 1
    np.save('labels.npy', np.array(labels_folders))


def train(learning_rate, learning_rate_decay, learning_rate_decay_step_size, batch_size, num_of_epochs, img_size, arch):
    # check device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # parameters
    RANDOM_SEED = 42
    N_CLASSES = 3

    # Load Data
    dataset = PoseDataset(
        csv_file='./labels.csv',
        img_size=img_size,
        transform=transforms.ToTensor())

    train_set, test_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # instantiate the model
    torch.manual_seed(RANDOM_SEED)

    if arch == 'simple':
        model = Classifier(N_CLASSES).to(DEVICE)

    elif arch == 'resnet':
        model = ResClassifier(N_CLASSES).to(DEVICE)

    else:
        print('model architecture not supported, you can use simple and resnet only!')
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=learning_rate_decay_step_size, gamma=learning_rate_decay)

    cross_entropy_loss_criterion = nn.CrossEntropyLoss()

    print('start training...')
    # start training
    model, optimizer, train_losses, valid_losses = training_loop(model,
                                                                cross_entropy_loss_criterion,
                                                                batch_size,
                                                                optimizer,
                                                                scheduler,
                                                                num_of_epochs,
                                                                train_loader,
                                                                test_loader,
                                                                DEVICE)


if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--do_data", action='store_true')
    argparser.add_argument("--do_train", action='store_true')
    argparser.add_argument('-dataset_path', '--path_to_dataset', type=str, help='path to dataset folder that contains folder for each label', default = '../dataset/')
    argparser.add_argument('-arch', '--architecture', type=str, help='model architecture,choose simple or resnet only!', default = 'simple')
    argparser.add_argument('-lr', '--learning_rate', type=float, help='starting learning rate', default=0.001)
    argparser.add_argument('-lr_decay', '--learning_rate_decay', type=float, help='decay rate of learning rate after step size', default=0.2)
    argparser.add_argument('-lr_decay_step_size', '--learning_rate_decay_step_size', type=int, help='number of epochs after which decay learning rate occurs', default=5)
    argparser.add_argument('-batch', '--batch_size', type=int, help='batch size', default=32)
    argparser.add_argument('-epochs', '--num_of_epochs', type=int, help='num of epochs', default=50)
    argparser.add_argument('-size', '--img_size', type=int, help='img size to downsample on before training', default=256)

    args = argparser.parse_args()

    if args.do_train:
        train(
            args.learning_rate,
            args.learning_rate_decay,
            args.learning_rate_decay_step_size,
            args.batch_size,
            args.num_of_epochs,
            args.img_size,
            args.architecture         
            )

    if args.do_data:
        generate_csv(args.path_to_dataset)

        
                                                                
                            



 

