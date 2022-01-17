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
#from BuildModel import BuildModel

class BuildModel:

    def train_model(self, data_dir,save_dir='',arch='vgg16',learning_rate=0.001,hidden_units=2048,epochs=10,gpu=False):
        train_data_transforms = transforms.Compose([ transforms.Resize(256),
                                        transforms.RandomRotation(10),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
        train_image_datasets =datasets.ImageFolder(data_dir+'/train', transform=train_data_transforms)
        train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)


        valid_data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

        valid_image_datasets =datasets.ImageFolder(data_dir+'/valid', transform=valid_data_transforms)
        valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size = 32, shuffle = True)

        model,criterion, optimizer=self.setup_network(arch,learning_rate,hidden_units)
        
        device = torch.device("cuda" if gpu else "cpu")
        model.to(device)

        #epochs = epochs
        print_every = 5
        #train_losses, test_losses = [], []
        #for i in keep_awake(range(5)):  
        #for e in keep_awake(range(epochs)):
        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(train_dataloaders):
                #print(inputs.shape)
                #newin=inputs.view(inputs.shape[0],-1)
                #print("Input",newin)
                #print(newin.shape)
                #print("Label",labels.shape())
                # Move input and label tensors to the GPU
                #print("running on ",device,ii)
                #print(".",end='')
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if (ii+1) % print_every == 0:
                    model.eval()
                    test_loss = 0
                    accuracy = 0
                    with torch.no_grad():
                        for inputs1, labels1 in valid_dataloaders:
                            inputs1, labels1 = inputs1.to(device), labels1.to(device)
                            
                            logps = model.forward(inputs1)
                            batch_loss = criterion(logps, labels1)
                            test_loss += batch_loss.item()
                            
                            
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels1.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            
                    #train_losses.append(running_loss/len(train_dataloaders))
                    #test_losses.append(test_loss/len(valid_dataloaders))
                            
                    #print(".")
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(test_loss/len(valid_dataloaders)),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(valid_dataloaders)))
                    running_loss = 0
                    model.train()
        

        model.class_to_idx = train_image_datasets.class_to_idx
        savedata = {'input_size':25088 ,
              'output_size': hidden_units,
              #'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict(),
               'class_to_idx':model.class_to_idx,
                'structure' :'vgg16',
            'classifier': model.classifier,
           }

        torch.save(savedata, save_dir+'/checkpoint.pth' if save_dir else 'checkpoint.pth')



    def setup_network(self,arch='vgg16', learning_rate=0.001,hidden_units=2048):
    
        inputs=0
        model=None
        if arch=='vgg16':
            model=models.vgg16(pretrained=True)
            inputs=25088
        elif arch=='alexnet':
            model=models.alexnet(pretrained = True)
            inputs=9216 
        elif arch=='vgg13':
            model=models.vgg13(pretrained=True)
            inputs=25088 
        else:
            print("Not yet configure model ",arch)

        #print(model)
        for param in model.parameters():
            param.requires_grad = False

        #print('INputs=',inputs,'hidden_units',hidden_units)
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(inputs, hidden_units)),
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(hidden_units, 256)),
                            ('relu2', nn.ReLU()),
                            ('fc3', nn.Linear(256,102)),
                            ('dropout1',nn.Dropout(p=0.5)),  #2048
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
        
        model.classifier = classifier
        #print(model)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        return model,criterion, optimizer