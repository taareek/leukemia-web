import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import pickle
import io
import base64
import feat_extractor

#defining pre-trained models path
cnn_path = "./models/lk_final_v2_20k_hsv_vgg19.pt"
ensemble_model_path = "./models/select_300.pkl"
feat_importance_path = "./models/train_importance_lr_clf.csv"

# reading feature importance 
f_importance = pd.read_csv(feat_importance_path)

# loading trained ensemble model 
model_in = open(ensemble_model_path, 'rb')
ensemble_model = pickle.load(model_in)

# defining class labels 
classes = ['ALL (Cancer)', 'HEM (Normal)']

###################### IMAGE CONVERSION ######################
# function to convert RGB image into HSV image 
def rgb_to_hsv(img):
  hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  return hsv_img


###################### IMAGE AUGMENTATION & TRANSFORMATION ######################
# data transform-> RGB to HSV, resize, to tensor, normalization and batch shape
def img_transformation(input_img):
  
  # defining mean and standard deviation for normalization
  mean = [0.5, 0.5, 0.5]
  std = [0.5, 0.5, 0.5]

  # defining transformation
  data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomRotation((0,180)),
        transforms.CenterCrop(150),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
                            ])
  
  img_bytes = input_img.file.read()
  img = Image.open(io.BytesIO(img_bytes))  # it needs then convert into numpy to work with opencv
  cv_img = np.array(img)

  # encoded image 
  encoded_string = base64.b64encode(img_bytes)
  bs64 = encoded_string.decode('utf-8')
  encoded_img = f'data:image/jpeg;base64,{bs64}'
  
  # convert input image RGB to CSV 
  hsv_converted = rgb_to_hsv(cv_img)
  # convert that image into PIL format for image transformation
  pil_img = Image.fromarray(hsv_converted)
  tensor_img = data_transforms(pil_img).unsqueeze(0)
  return cv_img, encoded_img, tensor_img


###################### FEATURE EXTRACTION & GRAD-CAM GENERATION #############################

def get_features_n_gradcam(input_img):
  # image transformation
  org_img, org_encoded_img, img = img_transformation(input_img)
  
  # forward pass 
  pred_img = feat_extractor.custom_model(img)
  img_features = feat_extractor.features['classifier[3]']

  index = pred_img.argmax(dim=1)
  ind_val = index.item()
  pred_img[:,ind_val].backward()

  # pull the gradients out of the model
  gradients = feat_extractor.custom_model.get_activations_gradient()

  # pool the gradients across the channels
  pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

  # get the activations of the last convolutional layer
  activations = feat_extractor.custom_model.get_activations(img).detach()

  # weight the channels by corresponding gradients
  for i in range(512):
      activations[:, i, :, :] *= pooled_gradients[i]

  # average the channels of the activations
  heatmap = torch.mean(activations, dim=1).squeeze()

  # relu on top of the heatmap
  heatmap = np.maximum(heatmap, 0)

  # normalize the heatmap
  heatmap /= torch.max(heatmap)
  # img_2 = cv2.imread(input_img)

  img_2 = rgb_to_hsv(org_img)
  img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

  heatmap = heatmap.numpy()
  heatmap = cv2.resize(heatmap, (img_2.shape[1], img_2.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

  superimposed_img = heatmap * 0.35 + img_2

  # encoded cam image 
  # Convert the OpenCV image to bytes
  retval, buffer = cv2.imencode('.jpg', superimposed_img)
  img_bytes = np.array(buffer).tobytes()

  # Encode the image bytes to base64
  encoded_string = base64.b64encode(img_bytes)
  bs64 = encoded_string.decode('utf-8')

  # Create the data URI
  cam_encoded_img = f'data:image/jpeg;base64,{bs64}'

  return org_encoded_img, cam_encoded_img, img_features


###################### FEATURE SELECTION ######################

def select_features(features, num_of_feat):
  # making train and test set using specific number of features based on feature importance that
    # we got from shapley values
    # a list for column names
  f_col = []
  for i in range(1, 1025):
    f_col.append("f_"+ str(i))
    
  feat_frame = pd.DataFrame(features, columns= f_col)
  train_x = feat_frame[f_importance.head(num_of_feat)["name"]]
  return train_x


###################### PREDICTION FUNCTION ######################
# get prediction results
def get_pred_results(img_features):

  feat_selected = select_features(img_features, 300)
  pred = ensemble_model.predict(feat_selected)
  pred_proba = ensemble_model.predict_proba(feat_selected)

  # class name 
  pred_class = classes[int(pred)]
 
  # probability
  proba = pred_proba[0][int(pred)]
  proba_round = round(proba * 100, 2)

  return pred_class, proba_round

###################### PREDICTION VIA AN INPUT IMAGE ######################
def get_prediction(input_img, is_api = False):
  
  # getting original image, grad-cam, image features 
  org_encoded_img, cam_encoded_img, img_features =  get_features_n_gradcam(input_img)

  # prediction via proposed ensemble model 
  pred_class, pred_proba = get_pred_results(img_features)

  pred_results = {
            "class_name": pred_class,
            "class_probability": pred_proba
        }
  
  # conditionally add image data to the result dictionary
  if not is_api:
      pred_results["org_encoded_img"] = org_encoded_img
      pred_results["cam_encoded_img"] = cam_encoded_img

  return pred_results

  
