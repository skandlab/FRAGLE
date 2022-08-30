import json
import pandas as pd
import csv
import os
import sys
import pickle
import numpy as np
from random import random
import math
import torch
import torchvision 
import torch.nn.functional as F  
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader  
from tqdm import tqdm  
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy


parentPath = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NN(nn.Module):
  def __init__(self, input_size=127):
    super(NN, self).__init__()

    self.NN_tumor = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(),
                            nn.Linear(64, 96), nn.ReLU(),
                            nn.Linear(96, 128), nn.ReLU(),
                            nn.Linear(128, 128), nn.ReLU())
    
    self.NN_blood = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(),
                            nn.Linear(64, 96), nn.ReLU(),
                            nn.Linear(96, 128), nn.ReLU(),
                            nn.Linear(128, 128), nn.ReLU())
    
    self.NN_join = nn.Sequential(nn.Linear(256, 64), nn.ReLU(),
                            nn.Linear(64, 96), nn.ReLU(),
                            nn.Linear(96, 128), nn.ReLU(),
                            nn.Linear(128, 1))
    
  def forward(self, x):
    x0, x1 = x[:, 0, :], x[:, 1, :]
    x0 = self.NN_tumor(x0)
    x1 = self.NN_blood(x1)
    x = torch.cat((x0, x1), 1)
    x = torch.squeeze(self.NN_join(x))
    return x

model_path = 'models/low_model.pt'
low_model = NN().to(device)
if device == 'cuda':
  low_model.load_state_dict(torch.load(model_path))
else:
  low_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


data_dict = {}
with open('Intermediate_Files/low_data.pkl', 'rb') as f:
  data_dict = pickle.load(f)
data = data_dict['samples']
sample_names = data_dict['meta_info']

max_arr = np.load('meta_info/max_arr.npy')
sums = np.sum(data, axis=2)
data = data/ sums[:, :, np.newaxis]
data = data/ max_arr


csv_list = []
low_model.eval()
with torch.no_grad():
  for i in range(data.shape[0]):
      dataX = torch.tensor(data[i]).type(torch.float)
      dataX = torch.unsqueeze(dataX, 0)
      score = low_model(dataX.to(device))
      csv_list.append([sample_names[i], np.round(score.item(),5)]) 
        
filePath = 'Intermediate_Files/low_predictions.csv'
my_df = pd.DataFrame(csv_list)
my_df.to_csv(filePath, index=False, header=['Sample_ID', 'Pred_TF']) 