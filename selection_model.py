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


parentPath = os.getcwd()
model_path = 'models/select_model.joblib'
select_model = load(model_path)


def make_prediction_csv():
    # omit the parts inside ### in the final code
    csv_list = []
    low_csv = 'Intermediate_Files/low_predictions.csv'
    high_csv = 'Intermediate_Files/high_predictions.csv'

    low_df = pd.read_csv(low_csv)
    low_pred_list = low_df.values.tolist()
    high_df = pd.read_csv(high_csv)
    high_pred_list = high_df.values.tolist()

    for low_sample in low_pred_list:
      low_sample_id = low_sample[0]
      for high_sample in high_pred_list:
        high_sample_id = high_sample[0]
        if (low_sample_id in high_sample_id) or (high_sample_id in low_sample_id):
          low_tf, high_tf = low_sample[-1], high_sample[-1]
          select_score = select_model.predict([[low_tf, high_tf]])[0]
          score = None
          if select_score == 1:
            score = high_tf
          elif select_score == 0:
            score = low_tf
          csv_list.append([low_sample_id, score]) 
          break
         
    filePath = 'Output/Predictions.csv'
    my_df = pd.DataFrame(csv_list)
    my_df.to_csv(filePath, index=False, header=['Sample_ID', 'Pred_TF'])

make_prediction_csv() 