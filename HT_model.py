import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision.models
from torch import Tensor
import json
import os
import pickle5 as pickle
import pandas as pd
import math
import sys


### Input
data_folder = sys.argv[1]
###


class MLPLayer(nn.Module):
    def __init__(self, hidden_dimension):
        super(MLPLayer, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.linear_layer = nn.Linear(
            self.hidden_dimension, self.hidden_dimension
            )
        self.bn = nn.BatchNorm1d(
            self.hidden_dimension
            )
    def forward(self, x: Tensor) -> Tensor:
        x_in = x
        x = self.linear_layer(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x + x_in
        return x
class CFDNAModel(nn.Module):
    """
    CFDNA Model to predict tumor fractions for given signals and scaleograms images.
    """
    def __init__(self, config, tumor_fraction_stats):
        super(CFDNAModel, self).__init__()
        # Define net parameters
        self.tumor_fraction_stats = tumor_fraction_stats
        if torch.cuda.is_available():
            self.dtypeFloat = torch.cuda.FloatTensor
            self.device = "cuda"
        else:
            self.dtypeFloat = torch.FloatTensor
            self.device = "cpu"
        #self.log_offset = config.log_offset
        self.smallest_tumor_fractions = config.smallest_tumor_fractions
        self.regression_dimension = config.regression_dimension
        self.classification_dimension = config.classification_dimension
        self.signal_dimension = 350
        self.imagenet_finetune = config.imagenet_finetune
        #self.motif_dimension = config.motif_dimension
        self.number_mlp_layers = config.number_mlp_layers
        self.dropout_rate = config.dropout_rate
        self.hidden_dimension = config.hidden_dimension
        #TODO: Util function to normalize tumor fractions
        self.zero_tumor_fractions_norm = (0 - self.tumor_fraction_stats["mean"])/self.tumor_fraction_stats["std"]
        self.smallest_tumor_fractions_norm = (self.smallest_tumor_fractions - self.tumor_fraction_stats["mean"])/self.tumor_fraction_stats["std"]
        self.one_tumor_fractions_norm = (1 -self. tumor_fraction_stats["mean"])/self.tumor_fraction_stats["std"]
        self.embedding_layer = nn.Linear(
           self.signal_dimension, self.hidden_dimension
           )
        # self.embedding_layer_2 = nn.Linear(
        #    4*self.hidden_dimension, self.hidden_dimension
        #    )
        self.embedding_layer_2_reg = nn.Linear(
            self.hidden_dimension, self.hidden_dimension
        )
        self.embedding_layer_3_reg = nn.Linear(
            self.hidden_dimension, self.hidden_dimension
        )
        self.embedding_layer_2_class = nn.Linear(
            self.hidden_dimension, self.hidden_dimension
        )
        self.embedding_layer_3_class = nn.Linear(
            self.hidden_dimension, self.hidden_dimension
        )
        self.bn = nn.BatchNorm1d(
            self.hidden_dimension
            )
        self.bn_2_reg = nn.BatchNorm1d(
            self.hidden_dimension
            )
        self.bn_3_reg = nn.BatchNorm1d(
            self.hidden_dimension
            )
        self.bn_2_class = nn.BatchNorm1d(
            self.hidden_dimension
            )
        self.bn_3_class = nn.BatchNorm1d(
            self.hidden_dimension
            )
        mlp_layers = []
        for layer in range(self.number_mlp_layers):
            mlp_layers.append(MLPLayer(self.hidden_dimension))
        self.mlp_layers = nn.ModuleList(mlp_layers)
        self.regression_layer = nn.Linear(
            self.hidden_dimension, self.regression_dimension
            )
        self.classification_layer = nn.Linear(
            self.hidden_dimension, self.classification_dimension
            )
        self.dropout_layer = nn.Dropout(
            self.dropout_rate
            )
    def forward(self, signal):
        # tumor_fraction, tumor_fraction_norm,
        # tumor_fraction_log, tumor_fraction_log_norm,
        # tumor_classification_label, classification_cw):
        """
        Args
            signal:
            signal_norm: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            scaleogram_image: Input edge distance matrix (batch_size, num_nodes, num_nodes)
            tumor_fraction: Input nodes (batch_size, num_nodes)
            tumor_fraction_norm: Input node coordinates (batch_size, num_nodes, node_dim)
            tumor_fraction_log:
            tumor_fraction_log_norm:
            tumor_classification_label:
            classification_cw: Class weights for tumor vs healhy classification loss
        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            # y_pred_nodes: Predictions for nodes (batch_size, num_nodes)
            loss: Value of loss function
            y_pred_reg_unnormalize
            y_pred_reg
            y_pred_class
            mse_loss
            mse_log_loss
            nll_loss
            unnormalize_mae
            unnormalize_mae_log
        """
        # Node and edge embedding
        x = self.embedding_layer(signal)
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
        x_class = self.embedding_layer_2_class(x)
        x_class = self.bn_2_class(x_class)
        x_class = F.relu(x_class)
        x_class = self.embedding_layer_3_class(x_class)
        x_class = self.bn_3_class(x_class)
        x_class = F.relu(x_class)
        y_pred_class = self.classification_layer(x_class)
        y_pred_reg = self.regression_layer(x_reg)
        return y_pred_reg, torch.squeeze(y_pred_class)


    def __unnormalize(self, pred_reg):
        return (
            self.tumor_fraction_stats["std"] * pred_reg
            + self.tumor_fraction_stats["mean"]
        )


    def __tumor_fraction_clipping(self, pred_reg, normalize = False):

        if normalize:
            pred_reg[pred_reg < self.zero_tumor_fractions_norm] = self.zero_tumor_fractions_norm
            pred_reg[pred_reg > self.one_tumor_fractions_norm] = self.one_tumor_fractions_norm

        else:
            pred_reg[pred_reg < 0] = 0
            pred_reg[pred_reg > 1] = 1

        return pred_reg


    def output(self, pred_reg, pred_prob):

        regression_unnormalize = self.__unnormalize(pred_reg)
        regression_unnormalize = self.__tumor_fraction_clipping(regression_unnormalize)

        classification_prediction = torch.sigmoid(torch.squeeze(pred_prob))

        regression_final = regression_unnormalize#*classification_prediction.unsqueeze(1)

        return regression_final, regression_unnormalize, classification_prediction




class Settings(dict):
    def __init__(self, config_dict):
        super().__init__()
        for key in config_dict:
            self[key] = config_dict[key]

    def __getattr__(self, attr):
        return self[attr]

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    __delattr__ = dict.__delitem__

def get_config(filepath):
    config = (Settings(json.load(open(filepath))))
    return config

config_train_path = 'meta_info/train.json'
config_train = get_config(config_train_path)



signals_stats = {}
with open('meta_info/signals_stats.pkl', 'rb') as f:
  signals_stats = pickle.load(f)

tumor_fractions_stats = {}
with open('meta_info/tumor_fractions.pkl', 'rb') as f:
  tumor_fractions_stats = pickle.load(f)



device = torch.device("cpu")
net = nn.DataParallel(CFDNAModel(config_train, tumor_fractions_stats))
checkpoint = torch.load("models/high_model.tar", map_location='cpu')
net.load_state_dict(checkpoint['net_state_dict'])



data_dict = {}
high_data_path = data_folder + 'high_data.pkl'
with open(high_data_path, 'rb') as f:
  data_dict = pickle.load(f)
data = data_dict['samples']
sample_names = data_dict['meta_info']



def dataProcess(signal):
  # max count normalize
  signal = signal + 1
  signal = signal / signal[np.argmax(signal)]
  # log
  signal = np.log10(signal)
  # running mean and std normalize
  filter_window = 32
  mean_signal, std_signal = [], []
  length_signal = len(signal)
  signal_pad = np.pad(signal, (filter_window//2, filter_window//2), mode = "edge")
  for i in range(0, length_signal):
    begin = i
    end = i + filter_window
    sliced_signal = signal_pad[begin:end]
    mean_signal.append(np.mean(sliced_signal))
    std_signal.append(np.std(sliced_signal))
  for i in range(len(std_signal)):
    if std_signal[i]==0:
      std_signal[i] = 1e-6 
  signal = np.array((signal - mean_signal)/std_signal, dtype="float32")
  
  # training stats mean and std columnwise normalize
  signal = (signal - signals_stats["mean"])/ signals_stats["std"]
  return signal



csv_list = []
net.eval()
with torch.no_grad():
  for i in range(data.shape[0]):
      dataX = dataProcess(data[i])
      dataX = torch.tensor(dataX).type(torch.float)
      dataX = torch.unsqueeze(dataX, 0)
      y_pred_reg, y_pred_class = net.forward(dataX)
      score, _, _ = net.module.output(y_pred_reg, y_pred_class)
      csv_list.append([sample_names[i], np.round(score.item(),5)])
        
filePath = data_folder + 'high_predictions.csv'
my_df = pd.DataFrame(csv_list)
my_df.to_csv(filePath, index=False, header=['Sample_ID', 'Pred_TF'])