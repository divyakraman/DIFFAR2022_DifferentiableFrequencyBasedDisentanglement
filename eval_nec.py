import sys
sys.path.append('~/miniconda3/pkgs')

import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

from dataset.necdrone_all import NECDataSet
from models.frameSampler import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
NUM_STEPS_STOP = 452
NUM_FRAMES = 8
INPUT_SIZE_SCALE = 2

DATA_DIRECTORY = ''
DATA_LIST_PATH = './dataset/NECDrone/test16.txt' 

model_link = ''
net = torch.load(model_link)
net = torch.nn.DataParallel(net).cuda() 


trainloader = data.DataLoader(
		NECDataSet(DATA_DIRECTORY, DATA_LIST_PATH, max_iters=NUM_STEPS_STOP,
					mean=IMG_MEAN, set='train', num_frames=NUM_FRAMES, input_size_scale=INPUT_SIZE_SCALE),
		batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
	
trainloader_iter = enumerate(trainloader)

accuracy = 0
top5_accuracy = 0

for i in range(0, NUM_STEPS_STOP):
	_, batch = trainloader_iter.__next__()
	uniform_frames, labels, all_frames = batch
	images = frameSampler(all_frames, uniform_frames, NUM_FRAMES, model_link)
	images = images.permute(0,2,1,3,4) #For I3D
	#images = images[0,:,:,:,:] #For spatial temporal baseline
	images = Variable(images).cuda()
	out_uniform = net(uniform_frames.permute(0,2,1,3,4))
	out_uniform = out_uniform[0].detach()
	out_uniform = out_uniform[:,:,0]
	out = net(images)
	out = out[0]
	out = out[:,:,0] #Only for X3D
	
	out = out.detach()
	out = out_uniform #+out
	outsort = torch.argsort(out,descending=True)
	out = torch.argmax(out)
	out = out.cpu().numpy()
	labels = labels.numpy()
	print("Iteration: ", i, ": ", outsort[0,:5],labels)

	if(out==labels):
		accuracy = accuracy+1

	out = outsort[0,:5].cpu().numpy()
	if labels in out:
		top5_accuracy = top5_accuracy + 1

	print("Accuracy is: ", (accuracy*100)/(i+1))


print("Accuracy is: ", (accuracy*100)/NUM_STEPS_STOP)	
print("Top 5 Accuracy is: ", (top5_accuracy*100)/NUM_STEPS_STOP)	
