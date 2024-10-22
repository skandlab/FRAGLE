# Importing libraries
# sklearn version 1.2.2 required: "pip install scikit-learn==1.2.2"
import json
import pandas as pd
import csv
import os
import sys
import pickle
import numpy as np
import random
import math
import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from joblib import dump, load


# Input and Output folder path
input_folder = sys.argv[1]
output_folder = sys.argv[2]

# loading meta_info for prediction
device = torch.device("cpu")
meta_info = {}
with open('meta_info/meta_info.pkl', 'rb') as f:
  meta_info = pickle.load(f)
sig_lens = meta_info['sig_lens']
feature_mean = meta_info['feature_mean']
feature_std = meta_info['feature_std']
TF_median = meta_info['TF_median']
TF_std = meta_info['TF_std']


# loading all three models
class MLPLayer(nn.Module):
    def __init__(self, hidden_dimension):
        super(MLPLayer, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.linear_layer = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        self.bn = nn.BatchNorm1d(self.hidden_dimension)
    def forward(self, x):
        x_in = x
        x = self.linear_layer(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x + x_in
        return x


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.regression_dimension = 1
        self.classification_dimension = 1
        self.signal_dimension = len(sig_lens)
        # self.signal_dimension = 350
        self.number_mlp_layers = 12
        self.dropout_rate = 0.2
        self.hidden_dimension = 256

        self.embedding_layer = nn.Linear(self.signal_dimension, self.hidden_dimension)
        self.bn = nn.BatchNorm1d(self.hidden_dimension)

        mlp_layers = []
        for layer in range(self.number_mlp_layers):
            mlp_layers.append(MLPLayer(self.hidden_dimension))
        self.mlp_layers = nn.ModuleList(mlp_layers)
        self.dropout_layer = nn.Dropout(self.dropout_rate)

        self.embedding_layer_2_reg = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        self.bn_2_reg = nn.BatchNorm1d(self.hidden_dimension)
        self.embedding_layer_3_reg = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        self.bn_3_reg = nn.BatchNorm1d(self.hidden_dimension)
        self.regression_layer = nn.Linear(self.hidden_dimension, self.regression_dimension)

        self.embedding_layer_2_class = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        self.bn_2_class = nn.BatchNorm1d(self.hidden_dimension)
        self.embedding_layer_3_class = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        self.bn_3_class = nn.BatchNorm1d(self.hidden_dimension)
        self.classification_layer = nn.Linear(self.hidden_dimension, self.classification_dimension)

    def forward(self, input):
        x = self.embedding_layer(input)
        x = self.bn(x)
        x = F.relu(x)

        for i in range(self.number_mlp_layers):
            x = self.mlp_layers[i](x)
        x = self.dropout_layer(x)

        x_reg = self.embedding_layer_2_reg(x)
        x_reg = self.bn_2_reg(x_reg)
        x_reg = F.relu(x_reg)
        x_reg = self.embedding_layer_3_reg(x_reg)
        x_reg = self.bn_3_reg(x_reg)
        x_reg = F.relu(x_reg)
        y_pred_reg = self.regression_layer(x_reg)

        x_class = self.embedding_layer_2_class(x)
        x_class = self.bn_2_class(x_class)
        x_class = F.relu(x_class)
        x_class = self.embedding_layer_3_class(x_class)
        x_class = self.bn_3_class(x_class)
        x_class = F.relu(x_class)
        y_pred_class = self.classification_layer(x_class)

        return torch.squeeze(y_pred_reg), torch.squeeze(y_pred_class)
    

def load_model(model_path):
    model = NN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

LT_model = load_model('Models/LT_model.pt')
HT_model = load_model('Models/HT_model.pt')
select_model = load('Models/select_model.pkl')


# data loading and feature generation
data_dict = {}
data_path = input_folder + 'data.pkl'
with open(data_path, 'rb') as f:
  data_dict = pickle.load(f)
data = data_dict['samples']
sample_names = data_dict['meta_info']

unit_frag_no = 1 * pow(10, 7)
for i in range(data.shape[0]):
    frag_no = np.sum(data[i])
    coverage = float( frag_no / unit_frag_no )
    if coverage<0.025:
        print(f'WARNING: too few reads in sample {sample_names[i]}')



data_X = np.copy(data)
data_X = data_X + 1
data_X = data_X / np.max(data_X, axis=1)[:, np.newaxis]
data_X = np.log10(data_X)
filter_window = 32
signal_pad = np.pad(data_X, ((0, 0), (filter_window//2, filter_window//2)), mode = "edge")
strided_view = np.lib.stride_tricks.sliding_window_view(signal_pad, filter_window, axis=1)
mean_signal = np.mean(strided_view, axis=2)
std_signal = np.std(strided_view, axis=2)
std_signal = np.where(std_signal == 0, 1e-6, std_signal)
data_X = np.array((data_X - mean_signal[:, :-1])/std_signal[:, :-1], dtype="float32")
data_X = (data_X - feature_mean)/ feature_std
data_X = data_X[:, sig_lens]


# LT and HT prediction
def predict_tf(model, model_type):
    global data_X
    csv_list = []
    csv_path = None
    model.eval()

    with torch.no_grad():
        for i in range(data_X.shape[0]):
            dataX = torch.tensor(data_X[i]).type(torch.float)
            dataX = torch.unsqueeze(dataX, 0)
            score, _ = model(dataX.to(device))
            if model_type=='LT':
                score = score.item()
                csv_path = output_folder + 'LT.csv'
            elif model_type=='HT':
                score = score.item() * TF_std + TF_median
                csv_path = output_folder + 'HT.csv'
            csv_list.append([sample_names[i], score])
            # csv_list.append([sample_names[i][0], sample_names[i][1], score])
    df = pd.DataFrame(csv_list)
    df.to_csv(csv_path, index=False, header=['Sample_ID', 'ctDNA_Burden'])
    # df.to_csv(csv_path, index=False, header=['Cohort', 'Sample_ID', 'ctDNA_Burden'])
    return csv_list

LT_preds = predict_tf(LT_model, 'LT')
HT_preds = predict_tf(HT_model, 'HT')

# Final Fragle Prediction
csv_list = []
for i in range(len(LT_preds)):
    sample_id = LT_preds[i][0]
    # cohort, sample_id = LT_preds[i][0], LT_preds[i][1]
    LT_TF, HT_TF = LT_preds[i][-1], HT_preds[i][-1]
    select_score = select_model.predict([[LT_TF, HT_TF]])[0]
    score = None
    if select_score == 1:
        score = HT_TF
    elif select_score == 0:
        score = LT_TF
    csv_list.append([sample_id, score])
    # csv_list.append([cohort, sample_id, score]) 

filePath = output_folder + 'Fragle.csv'
df = pd.DataFrame(csv_list)
df.to_csv(filePath, index=False, header=['Sample_ID', 'ctDNA_Burden'])
# df.to_csv(filePath, index=False, header=['Cohort', 'Sample_ID', 'ctDNA_Burden'])
            
