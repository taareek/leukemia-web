## Leukemia-Web
This repository contains a web application for the automated diagnosis of Leukemia Cancer. We have employed a hybrid approach by integrating machine learning and deep learning techniques to develop this Computer-Aided Diagnosis (CAD) system. FastAPI was employed as the framework to create the proposed model as an API in this project. It represents the implementation of our proposed model, as detailed in the paper titled 'Addressing Label Noise in Leukemia Image Classification Using Small Loss Approach and pLOF with Average-Weighted Ensemble.


## Getting Started
To run this project, you must have a Python environment set up on your local machine. It is recommended to create a virtual environment specifically for this project and install the required dependencies listed in the `requirements.txt` file. You can clone this repository into a folder within your newly created environment by using the following command:  
```
git clone https://github.com/taareek/leukemia-web.git
```
After that, you just need to execute the following command from the command prompt to run this application on your local server:  
```
uvicorn main:app --reload
``` 
Follow the output URL to access the web application.

### Test System with Input Images
To facilitate real-time evaluation of this system, we have provided a set of test data. You can access the test images through the following link:

* [test image folder](https://drive.google.com/drive/folders/17KnWCdDVS2kcBu1nuxjntVC3MrnRPYLK?usp=sharing/)

All of these images serve as test data and are completely unseen by the model. There are two subfolders: **ALL** and **HEM**, representing the classes.**HEM** contains healthy cells, while **ALL** contains leukemia cells. This system generates an output Grad-CAM image and predicts the class along with its corresponding probability for a given input test image.


