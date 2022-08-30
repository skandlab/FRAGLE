import pickle
import os
import pysam
import numpy as np


parentPath = os.getcwd()

contig_windows = {}
with open('meta_info/contig_windows.pkl', 'rb') as handle:
  contig_windows = pickle.load(handle)

our_contigs = []
for i in range(1, 23):
  our_contigs.append('chr'+str(i))


# Helper functions for sample generation
def init_len_dict():
  len_dict = {}
  for i in range(1, 401):
    len_dict[i] = 0
  return len_dict

def dict_to_list(dictt):
  lst = []
  for i in range(1, 401):
    lst.append(dictt[i])
  return lst

# traversing bam file and filling up the windows based on length frequency
def sample_generation(bamPath): 
  sample = []
  len_dict = init_len_dict()
  pybam = pysam.AlignmentFile(bamPath, "rb")
  for contig in our_contigs: # genome
    range_list = contig_windows[contig]
    for lst in range_list: # contig
      start, end = lst[0], lst[1]
      for read in pybam.fetch(contig, start, end): # window
        length = read.template_length
        if length<=0 or length>400:
          continue
        else:
          len_dict[length] += 1
      sample.append(dict_to_list(len_dict))
      len_dict = init_len_dict()
  
  sample_arr = np.array(sample).astype(np.int16)
  return sample_arr


os.chdir('bam_files')
sample_cnt = 0
for folder in os.listdir():
  sample_cnt += 1
data = np.zeros((sample_cnt, 289, 400))

i = 0
sample_names = []
for folder in os.listdir():
  data[i] = sample_generation(folder + '/sample.bam')
  sample_names.append(folder)
  i += 1

no_read_windows = [13, 159, 209, 221, 232, 233, 284]
read_windows = []
for i in range(289):
  if i not in no_read_windows:
    read_windows.append(i)
data = data[:, read_windows, :]


os.chdir(parentPath)
sig_dict = {}
with open('meta_info/sig_bins.pickle', 'rb') as handle:
  sig_dict = pickle.load(handle)
sig_lens = sig_dict['sig_lens']

gene_bin_dict = {}
with open('meta_info/gene_bin.pkl', 'rb') as handle:
  gene_bin_dict = pickle.load(handle)
tumor_bins = gene_bin_dict['cancer']
blood_bins = gene_bin_dict['blood']


high_data = np.sum(data[:, :, 50:], axis=1)

low_data = data[:, :, 50:]
low_data = low_data[:, :, sig_lens]
low_tumor = np.expand_dims( np.sum(low_data[:, tumor_bins, :], axis=1), axis=1)
low_blood = np.expand_dims( np.sum(low_data[:, blood_bins, :], axis=1), axis=1)
low_data = None
low_data = np.concatenate((low_tumor, low_blood), axis=1)


low_dict = {}
low_dict['samples'] = low_data
low_dict['meta_info'] = sample_names
os.chdir(parentPath) 
with open('Intermediate_Files/low_data.pkl', 'wb') as f:
  pickle.dump(low_dict, f)


high_dict = {}
high_dict['samples'] = high_data
high_dict['meta_info'] = sample_names
os.chdir(parentPath) 
with open('Intermediate_Files/high_data.pkl', 'wb') as f:
  pickle.dump(high_dict, f)