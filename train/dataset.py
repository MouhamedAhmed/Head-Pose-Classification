import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
from skimage.transform import resize

'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''

class PoseDataset(Dataset):

    def __init__(self, csv_file, img_size, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
        self.img_size = img_size
        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        img_path = self.annotations.iloc[index,0]
        img = io.imread(img_path)
        img = resize(img, (self.img_size, self.img_size), anti_aliasing=True)
        label = torch.tensor(int(self.annotations.iloc[index,1]))
        return img, label

    # we can call len(dataset) to return the size
    def __len__(self):
        return len(self.annotations)

