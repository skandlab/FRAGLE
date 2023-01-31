import json
import pandas as pd
import csv
import os
import sys
import pickle
import numpy as np
from random import random
import math
import copy
from sklearn.svm import SVC
from joblib import dump, load
import sys


### Input
data_folder = sys.argv[1]
###


model_path = 'models/select_model.joblib'
select_model = load(model_path)



def make_prediction_csv():
    # omit the parts inside ### in the final code
    detail_csv_list = []
    low_csv = data_folder + 'low_predictions.csv'
    high_csv = data_folder + 'high_predictions.csv'

    low_df = pd.read_csv(low_csv)
    low_pred_list = low_df.values.tolist()
    high_df = pd.read_csv(high_csv)
    high_pred_list = high_df.values.tolist()

    for i in range(len(low_pred_list)):
      sample_id = low_pred_list[i][0]
      low_tf, high_tf = low_pred_list[i][-1], high_pred_list[i][-1]
      select_score = select_model.predict([[low_tf, high_tf]])[0]
      score = None
      if select_score == 1:
        score = high_tf
      elif select_score == 0:
        score = low_tf
      detail_csv_list.append([sample_id, score]) 

    filePath = data_folder + 'Predictions.csv'
    my_df = pd.DataFrame(detail_csv_list)
    my_df.to_csv(filePath, index=False, header=['Sample_ID', 'Fragle_Prediction'])

make_prediction_csv() 