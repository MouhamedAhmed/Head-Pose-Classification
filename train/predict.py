import torch
import numpy as np
from skimage import io
from skimage.transform import resize

class Predictor():

    def __init__(self, model_path, labels_path):
        # check device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load model
        self.model = torch.load('./model.pkl').to(self.device)
        self.model.eval()

        # load labels file
        self.labels = np.load(labels_path)

    def predict(self, image_path):
        # read image
        img_path = image_path
        img = io.imread(img_path)

        # resize
        img = resize(img, (256, 256), anti_aliasing=True)

        # add axis and convert to tensor
        img_tensor = torch.unsqueeze(torch.Tensor(img), 0)

        # transpose
        img_tensor = torch.Tensor(img_tensor).to(self.device).transpose(1,3).transpose(2,3).float()

        #predict
        y_hat = self.model(img_tensor)
        predicted_label = torch.argmax(y_hat)
        label = predicted_label.item()
        return self.labels[label]

