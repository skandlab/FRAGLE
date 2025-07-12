import os
import pickle
import subprocess
import sys
import numpy as np

input_folder = sys.argv[1]
output_folder = sys.argv[2]
CPU = int(sys.argv[3])
bin_locations = sys.argv[4]  # Already resolved by main.py

# Resolve sample_feature_generation.py path using FRAGLE_HOME
fragle_home = os.environ.get("FRAGLE_HOME", ".")
sample_feature_script = os.path.join(fragle_home, "sample_feature_generation.py")

sample_cnt = 0
for file_ in os.listdir(input_folder):
    if file_.endswith(".bam"):
        sample_cnt += 1

data = np.zeros((sample_cnt, 350))
data_meta = []

i = 0
for file_ in os.listdir(input_folder):
    if file_.endswith(".bam"):
        bamfile = os.path.join(input_folder, file_)
        output_path = os.path.join(output_folder, f"{file_[:-4]}.npy")
        print(i + 1, file_)
        data_meta.append(file_[:-4])
        command = f"python {sample_feature_script} {bamfile} {output_path} {CPU} {bin_locations}"
        subprocess.run(command, shell=True, check=False)
        data[i] = np.load(output_path)
        os.remove(output_path)
        i += 1

data_dict = {"samples": data, "meta_info": data_meta}
output_file = os.path.join(output_folder, "data.pkl")
with open(output_file, "wb") as f:
    pickle.dump(data_dict, f)
