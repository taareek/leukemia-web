import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import pickle
import io

#defining pre-trained models path
cnn_path = "./models/lk_final_v2_20k_hsv_vgg19.pt"

# getting vgg-19 model
vgg19_model = models.vgg19(weights='IMAGENET1K_V1')
# change the number of classes
vgg19_model.classifier[2] = nn.Dropout(p= 0.5, inplace= False)
vgg19_model.classifier[3] = nn.Linear(4096, 1024)
vgg19_model.classifier[5] = nn.Dropout(p= 0.25, inplace= False)
vgg19_model.classifier[6] = nn.Linear(1024, 2)

# loading pre-trained weights
vgg19_model.load_state_dict(torch.load(cnn_path, map_location=torch.device('cpu')))


# defining custom model
class VGG_19_Extractor(nn.Module):
    def __init__(self):
        super(VGG_19_Extractor, self).__init__()

        # get the pretrained VGG-19 network
        self.vgg = vgg19_model

        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]

        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # get the classifier of the model
        self.classifier = self.vgg.classifier

        # placeholder for the gradients
        self.gradients = None


    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)
    
# define hook
# hook is use to get features from expected layer
features = {}

def get_features(name):
  def hook(model, input, output):
    if torch.cuda.is_available():
      features[name] = output.cpu().detach().numpy()
    else:
      features[name] = output.detach().numpy()
  return hook

# initialize the Custom model
custom_model = VGG_19_Extractor()

# register hook to get 1024 features
custom_model.classifier[4].register_forward_hook(get_features('classifier[3]'))

# evaluation mode
custom_model.eval()
