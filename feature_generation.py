import argparse
import os
import pathlib
import subprocess
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, nargs="*", required=True, help='Path(s) to one or more `.bam` files.')
parser.add_argument('--output', type=str, required=True, help='Output folder path where the Fragle predictions and processed features can be found')
parser.add_argument('--cpu', type=int, required=True, help='Number of CPUs to use for parallel processing of bam files')
parser.add_argument('--bin_locations', type=str, required=True, help='Path to file with bin locations.')
args = parser.parse_args()

sample_cnt = len(args.input)

if not os.path.exists(args.output):
    os.mkdir(args.output)

data = np.zeros((sample_cnt, 350))
data_meta = []

i = 0
for file_ in args.input:
    filename = pathlib.Path(file_).name
    if filename[-4:] != ".bam":
        raise ValueError(
            "Feature Generation: Input file path did not end with '.bam'."
        )  # Shouldn't happen
    output_path = f"{args.output}/{filename[:-4]}.npy"
    print(i + 1, file_)
    data_meta.append(file_[:-4])
    command = f"python sample_feature_generation.py {file_} {output_path} {args.cpu} {args.bin_locations}"
    subprocess.run(command, shell=True)
    data[i] = np.load(output_path)
    os.remove(output_path)
    i += 1

data_dict = {}
data_dict["samples"] = data
data_dict["meta_info"] = data_meta
output_file = f"{args.output}/data.pkl"
with open(output_file, "wb") as f:
    pickle.dump(data_dict, f)
