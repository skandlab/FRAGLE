import os
import subprocess
import numpy as np
import pandas as pd
import pickle
import copy, sys


input_folder = sys.argv[1]
output_folder = sys.argv[2]
CPU = sys.argv[3]
bin_locations = sys.argv[4]
if os.path.exists(output_folder)==False:
  os.mkdir(output_folder)

sample_cnt = 0
for file_ in os.listdir(input_folder):
  if file_.endswith('.bam'):
    sample_cnt += 1
data = np.zeros((sample_cnt, 350))
data_meta = []

i = 0
for file_ in os.listdir(input_folder):
  if file_.endswith('.bam'):
    bamfile = f'{input_folder}/{file_}'
    output_path = f'{output_folder}/{file_[:-4]}.npy'
    print(i+1, file_)
    data_meta.append(file_[:-4])
    command = f'python sample_feature_generation.py {bamfile} {output_path} {CPU} {bin_locations}'
    subprocess.run(command, shell=True)
    data[i]= np.load(output_path)
    os.remove(output_path)
    i += 1

data_dict = {}
data_dict['samples'] = data
data_dict['meta_info'] = data_meta
output_file = f'{output_folder}/data.pkl' 
with open(output_file, 'wb') as f:
  pickle.dump(data_dict, f)
