import argparse
import BuildModel as bm

argparser=argparse.ArgumentParser()
argparser.add_argument("data_dir")
argparser.add_argument('--save_dir',help='Set directory to save checkpoints', type=str,default="")
argparser.add_argument('--arch',help='Choose architecture',  type=str,default="vgg16")
argparser.add_argument('--learning_rate', help='learning rate', type=float,default=0.001)
argparser.add_argument('--hidden_units', help='hidden units', type=int,default=2048)
argparser.add_argument('--epochs',help='Epochs', type=int,default=10)
argparser.add_argument('--gpu',action="store_true", help='Use GPU for training')


args=argparser.parse_args()

#print(args.data_dir)
#print("###########"*2,"TRAINING WITH ")
#print(args)
buil=bm.BuildModel()
buil.train_model(args.data_dir,args.save_dir,args.arch,args.learning_rate,args.hidden_units,args.epochs,args.gpu)


# import torch
# from torchvision import datasets, transforms,models
# from torch import nn

# model = models.vgg13 (pretrained=True)
# print(model)