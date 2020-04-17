import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
from torchvision.models import vgg19
from torchvision import transforms, utils
from torchvision import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import cv2
import os
from PIL import Image

class SimpleCNN(torch.nn.Module):
        
    #Our batch shape for input x is (4, 256, 256)
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
       
        # 256 x 256 x 4
        #Input channels = 4, output channels = 8
        self.conv1 = torch.nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 128 x 128 x 8
        #Input channels = 8, output channels = 16
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 64 x 64 x 16
        #Input channels = 16, output channels = 32
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 32 x 32 x 32
        #Input channels = 32, output channels = 64
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(32 * 32 * 32, 256)
        
        #64 input features, 10 output features for our 2 defined classes
        self.fc2 = torch.nn.Linear(256, 64)
        
        self.fc3 = torch.nn.Linear(64, 2)
        
        self.dropout = torch.nn.Dropout(0.01)
        
        self.softmax = torch.nn.Softmax()

        self.gradients = None
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (4, 256, 256) to (8, 256, 256)
        x = F.relu(self.conv1(x))
        
        #Size changes from (8, 256, 256) to (8, 128, 128)
        x = self.pool1(x)
        
        #Size changes from (8, 128, 128) to (16, 128, 128)
        x = F.relu(self.conv2(x))
        
        #Size changes from (16, 128, 128) to (16, 64, 64)
        x = self.pool2(x)
        
        #Size changes from (16, 64, 64) to (32, 64, 64)
        x = F.relu(self.conv3(x))
        
        #Size changes from (32, 64, 64) to (32, 32, 32)
        x = self.pool3(x)
        
        h = x.register_hook(self.activations_hook)

        #Reshape data to input to the input layer of the neural net
        #Size changes from (32, 32, 32) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 32 * 32 * 32)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 32768) to (1, 64)
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        
        #x = self.dropout(x)
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 2)
        x = self.fc3(x)
        
        x = self.softmax(x)
        #return(F.log_softmax(x, dim=1))
        return x

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        #Computes the activation of the first convolution
        #Size changes from (4, 256, 256) to (8, 256, 256)
        x = F.relu(self.conv1(x))
        
        #Size changes from (8, 256, 256) to (8, 128, 128)
        x = self.pool1(x)
        
        #Size changes from (8, 128, 128) to (16, 128, 128)
        x = F.relu(self.conv2(x))
        
        #Size changes from (16, 128, 128) to (16, 64, 64)
        x = self.pool2(x)
        
        #Size changes from (16, 64, 64) to (32, 64, 64)
        x = F.relu(self.conv3(x))

        return x


def gradcam(img_name):
    
    basename = os.path.basename(os.path.splitext(img_name)[0])
    image = Image.open(img_name)
    image = image.resize((256, 256), Image.AFFINE)
    image = np.array(transform(image))
    image = image.reshape((1, 4, 256, 256))
    image = Variable(torch.from_numpy(image))

    pred = model(image)
    #print(pred.argmax(dim=1))
    print(pred.data)

    pred[:, 1].backward()
    gradients = model.get_activations_gradient()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = model.get_activations(image).detach()

    for i in range(32):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    heatmap = np.array(heatmap)

    img = cv2.imread(img_name)
    #img = cv2.resize(img, (256, 256))
    img[:,:,0] = img[:,:,1]
    img[:,:,2] = img[:,:,1]
    cv2.imwrite('./gradcam_images/' + basename + '_original.jpg', img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite('./gradcam_images/' + basename + '_' + '%.2f' % (pred[0][1].item()) + '_map.jpg', superimposed_img)

model = SimpleCNN()
model.load_state_dict(torch.load('../cnn/results/rs_500/cnn_random_after_training'))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5,0.5), (0.5,0.5,0.5,0.5))])
#img_name = sys.argv[1]

df = pd.read_csv("~/solar-flares/datasets/all_labels_sorted_small_dataset.csv", sep=",", header='infer')
df['flare'] = df.loc[:,'M_flare_in_24h'] + df.loc[:, 'X_flare_in_24h']
df = df[df.flare != 0]

for i, row in df.iterrows():
    gradcam(row['filename'])


    #img_name = '/srv/data/varad/data/all_images/hmi.sharp_cea_720s.8.20100505_150000_TAI.png'
    #img_name = '/srv/data/varad/data/all_images/hmi.sharp_cea_720s.377.20110214_060000_TAI.png'
    #img_name = '/srv/data/varad/data/all_images/hmi.sharp_cea_720s.5298.20150311_160000_TAI.png'
    #img_name = '/srv/data/varad/data/all_images/hmi.sharp_cea_720s.892.20110922_090000_TAI.png'
