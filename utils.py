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
from collections import Counter

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

# function to convert RGB image into HSV image 
def rgb_to_hsv(img):
#   img = cv2.imread(img_path)
#   org_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  return hsv_img

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
  # cv_img is the original image

  # encoded image 
  encoded_string = base64.b64encode(img_bytes)
  bs64 = encoded_string.decode('utf-8')
  encoded_img = f'data:image/jpeg;base64,{bs64}'
  
  # convert input image RGB to CSV 
  hsv_converted = rgb_to_hsv(cv_img)
  # convert that image into PIL format for image transformation
  pil_img = Image.fromarray(hsv_converted)
  tensor_img = data_transforms(pil_img).unsqueeze(0)
  return cv_img, tensor_img, encoded_img



################### ensemble augmentation predict ###############################
def mod_img_transformation(input_img):
  
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
    
  hor_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
                            ])
    
  ver_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
                            ])
    
  rot_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation((0,180)),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
                            ])

  crop_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(150),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
                            ])
  
  img_bytes = input_img.file.read()
  img = Image.open(io.BytesIO(img_bytes))  # it needs then convert into numpy to work with opencv
  cv_img = np.array(img)
  # cv_img is the original image

  # encoded image 
  encoded_string = base64.b64encode(img_bytes)
  bs64 = encoded_string.decode('utf-8')
  encoded_img = f'data:image/jpeg;base64,{bs64}'
  
  # convert input image RGB to CSV 
  hsv_converted = rgb_to_hsv(cv_img)
  
  # convert that image into PIL format for image transformation
  pil_img = Image.fromarray(hsv_converted)

  # applying transform and data augmentation 
  tensor_img = data_transforms(pil_img).unsqueeze(0)

  # horizontal
  hor_tensor = hor_transforms(pil_img).unsqueeze(0)

  # vertical 
  ver_tensor = ver_transforms(pil_img).unsqueeze(0)

  # rotation
  rot_tensor = rot_transforms(pil_img).unsqueeze(0)

  # crop 
  crop_tensor = crop_transforms(pil_img).unsqueeze(0)

  return cv_img, encoded_img, tensor_img, hor_tensor, ver_tensor, rot_tensor, crop_tensor

##########################################################################################
file_path = "./images/test/"
def test_image(input_img):
  img_bytes = input_img.file.read()
  img = Image.open(io.BytesIO(img_bytes))
  cv_img = np.array(img)
#   cv2.imwrite(file_path + 'org.png', cv_img)
  plt.imsave(file_path + 'org.png', cv_img)
  # convert input image RGB to CSV 
  hsv_converted = rgb_to_hsv(cv_img)
  plt.imsave(file_path + 'conv.png', hsv_converted)
  return True

############################################################################################
cam_path = "./images/cam/"

def get_features_n_gradcam(input_img):
  # org_img, img, org_encoded_img = img_transformation(input_img)

  # when majority voting approach used
  org_img, org_encoded_img, img, hor_img, ver_img, rot_img, crop_img = mod_img_transformation(input_img)   # take image path as input
  print("transformation done")
  
  # forward pass 
  pred_img = feat_extractor.custom_model(img)
  img_features = feat_extractor.features['classifier[3]']
  print("original feature extracted !")

  # # when majority voting approach used
  pred_hor = feat_extractor.custom_model(hor_img)
  hor_features = feat_extractor.features['classifier[3]']
  print("hor feature extracted !")

  pred_ver = feat_extractor.custom_model(ver_img)
  ver_features = feat_extractor.features['classifier[3]']
  print("ver feature extracted !")

  pred_rot = feat_extractor.custom_model(rot_img)
  rot_features = feat_extractor.features['classifier[3]']
  print("rot feature extracted !")

  pred_crop = feat_extractor.custom_model(crop_img)
  crop_features = feat_extractor.features['classifier[3]']
  print("crop feature extracted !")

  # get image features
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

  # superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
  # plt.imsave(cam_path + 'org_try.png', org_img)
  # # cv2.imwrite(cam_path + 'hsv.png', img_2)
  # cv2.imwrite(cam_path + 'cam_try.png', superimposed_img)
  # print('shape of features: ',img_features.shape)

  # encoded cam image 
  # Convert the OpenCV image to bytes
  retval, buffer = cv2.imencode('.jpg', superimposed_img)
  img_bytes = np.array(buffer).tobytes()

  # Encode the image bytes to base64
  encoded_string = base64.b64encode(img_bytes)
  bs64 = encoded_string.decode('utf-8')

  # Create the data URI
  cam_encoded_img = f'data:image/jpeg;base64,{bs64}'

  return org_encoded_img, cam_encoded_img, img_features, hor_features, ver_features, rot_features, crop_features


#################################################################

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


# get prediction results
def get_pred_results(img_features):

  feat_selected = select_features(img_features, 300)
  pred = ensemble_model.predict(feat_selected)
  pred_proba = ensemble_model.predict_proba(feat_selected)

  # class name 
  # pred_class = classes[int(pred)]

  # probability
  proba = pred_proba[0][int(pred)]
  proba_round = round(proba * 100, 2)

  return pred, proba_round

# function to find most frequent value in a list 
def most_appeared(a_list):
  occurance_count = Counter(a_list)
  freq_value = occurance_count.most_common(1)[0][0]
  frequency = occurance_count.most_common(1)[0][1]
  return freq_value, frequency

# get indexes for most frequent value
def get_indexes(freq_value, class_list):
  index_value = [index for index in range(len(class_list)) if class_list[index] == freq_value]
  return index_value

# find max probability 
def find_max_proba(index_list, proba_list, max_proba=0):
  for i in range(len(index_list)):
    k = index_list[i]
    if(proba_list[k] >= max_proba):
      max_proba = proba_list[k]
  return max_proba


def get_prediction(input_img, is_api = False):
  
  # org_encoded_img, img_features, cam_encoded_img =  get_features_n_gradcam(input_img)

  org_encoded_img, cam_encoded_img, img_features, hor_features, ver_features, rot_features, crop_features = get_features_n_gradcam(input_img)
  
  
  # list to keep predicted classes and probabilities
  pred_classes = []
  pred_probas = []

  ## original
  org_pred_class, org_pred_proba = get_pred_results(img_features)
  pred_classes.append(int(org_pred_class))
  pred_probas.append(org_pred_proba)
  print("pred done")

  # horizontal
  hor_pred_class, hor_pred_proba = get_pred_results(hor_features)
  pred_classes.append(int(hor_pred_class))
  pred_probas.append(hor_pred_proba)
  print("pred done")

  # vertical
  ver_pred_class, ver_pred_proba = get_pred_results(ver_features)
  pred_classes.append(int(ver_pred_class))
  pred_probas.append(ver_pred_proba)
  print("pred done")

  # rotation
  rot_pred_class, rot_pred_proba = get_pred_results(rot_features)
  pred_classes.append(int(rot_pred_class))
  pred_probas.append(rot_pred_proba)
  print("pred done")

  # crop
  crop_pred_class, crop_pred_proba = get_pred_results(crop_features)
  pred_classes.append(int(crop_pred_class))
  pred_probas.append(crop_pred_proba)
  print("pred done")

  # print(pred_classes)
  # print(pred_probas)

  # most predicted class  
  mostly_predicted, frequency = most_appeared(pred_classes)
  print("mostly pred done")

  # get the index numbers of most appeared class  
  mostly_predicted_indices = get_indexes(mostly_predicted, pred_classes)
  print("indices found")

  # # get maximum probability
  max_probability = find_max_proba(mostly_predicted_indices, pred_probas)
  print("probability found  found")

  pred_class = classes[mostly_predicted]

  pred_results = {
            "class_name": pred_class,
            "class_probability": max_probability
        }
  
  # conditionally add image data to the result dictionary
  if not is_api:
      pred_results["org_encoded_img"] = org_encoded_img
      pred_results["cam_encoded_img"] = cam_encoded_img

  return pred_results

  
