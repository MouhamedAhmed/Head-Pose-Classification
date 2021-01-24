import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

class Classifier(nn.Module):    
    def __init__(self,n_classes):
        super(Classifier,self).__init__()
        self.n_classes = n_classes
        
        #256
        self.CNN1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding = 1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        #127
        self.CNN2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        #62
        self.CNN3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        #30
        self.CNN4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        #14
        self.CNN5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # 7
        self.CNN6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        #3
        self.CNN7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),

        )

        self.FC1 = nn.Sequential(
            nn.Linear(in_features=32, out_features=self.n_classes),
        )
        
    def forward(self,x):
       x = self.CNN1(x)
       x = self.CNN2(x)
       x = self.CNN3(x)
       x = self.CNN4(x)
       x = self.CNN5(x)
       x = self.CNN6(x)
       x = self.CNN7(x).squeeze()
       x = self.FC1(x)
       return x


class ResClassifier(nn.Module):    
    def __init__(self, n_classes):
        super(ResClassifier,self).__init__()
        self.n_classes = n_classes
        self.resnet18 = models.resnet18()
        self.resnet18.fc = nn.Linear(in_features=512, out_features=self.n_classes)
                
    def forward(self,x):
        x = self.resnet18(x)
        return x
