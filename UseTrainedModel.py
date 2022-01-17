
import torch
from torchvision import datasets, transforms,models
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
import time
#from workspace_utils import keep_awake
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from BuildModel import BuildModel
import os
import json

class UseTrainedModel:


    def make_predictions(self,imagePath,checkpoint=None,top_k=3,category_names=None,gpu=False):
        model=self.load_model(checkpoint)
        top_p = self.predict(imagePath,model,top_k,gpu)
        #print('top_p',top_p)
        #print('top_class',top_class)
        probTop = np.array(top_p[0][0])[0]
        nameIdx=np.array(top_p[1][0])[0]+1
        #name=np.array(top_class[0])
        #pred_idx = top_p.cpu().numpy().argmax()
        #print('pred_idx',probTop)
        
        #print('Name Index',nameIdx)

        with open(category_names if category_names else 'cat_to_name.json', 'r') as f:
            self.cat_to_name = json.load(f)
    
        name='Name: '+self.cat_to_name[str(nameIdx)]+' Probability: '+str(probTop)
        print(name)
        return name


    def load_model(self,checkpointPath=None):
        buil=BuildModel()
        if not os.path.exists(checkpointPath):
            checkpointPath=None

        checkpoint = torch.load( checkpointPath if checkpointPath else 'checkpoint.pth')
        structure = checkpoint['structure']
        hidden_size=checkpoint['output_size']
        model,criterion,optimizer=buil.setup_network(structure,0.01,hidden_size)
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
       
        return model 

    def predict(self,image_path, model, topk=5,gpu=False):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        
        '''
        device = torch.device("cuda" if gpu else "cpu")
        #print("Using device ",device)
        model.to(device)
        model.eval()
        
        img_tensor =self.process_image(image_path)
        img_tensor=img_tensor.unsqueeze_(0)
        #image_npy=image_npy.numpy()
        #torchimage = torch.from_numpy(np.array([image_npy])).float()

        with torch.no_grad():
            logpsh = model.forward(img_tensor.to(device))
            
        return torch.exp(logpsh).data.topk(topk,dim=1)


    def process_image(self,image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        pil_image=Image.open(image)
        
        image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return image_transform(pil_image)


    def imshow(self,image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))
        
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        return ax