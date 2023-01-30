import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.utils import make_grid
import time
import copy
from torchvision import datasets, models, transforms
from PIL import Image
def load_image(image_file):
    img = Image.open(image_file)
    return img

label_map={
    0:"Chickenpox",
    1:"Measles",
    2:"Monkeypox",
    3:"Normal"
}
classes = ('Chickenpox', 'Measles', 'Monkeypox', 'Normal')
PATH = 'C:/Users/sajan/OneDrive/Desktop/Main Project/Project/resnet18_net.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.Resize((64,64)),
                                     transforms.ToTensor()])

def load_model():
    '''
    load a model 
    by default it is resnet 18 for now
    '''

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.to(device)

    model.load_state_dict(torch.load(PATH,map_location=device))
    model.eval()
    return model

def image_loader(image_name):
    """load image, returns cuda tensor"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    picture = Image.open(image_name)
    image = data_transform(picture)
    images=image.reshape(1,1,64,64)
    new_images = images.repeat(1, 3, 1, 1)
    return new_images
    
def predict(model, image_name):
    '''
    pass the model and image url to the function
    Returns: a list of pox types with decreasing probability
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    picture = Image.open(image_name)
    image = data_transform(picture)
    images=image.reshape(1,1,64,64)
    new_images = images.repeat(1, 3, 1, 1)

    outputs=model(new_images)

    _, predicted = torch.max(outputs, 1)
    ranked_labels=torch.argsort(outputs,1)[0]
    probable_classes=[]
    for label in ranked_labels:
        probable_classes.append(classes[label.numpy()])
    probable_classes.reverse()
    return probable_classes