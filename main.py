import subprocess
import os

P = subprocess.run("python data_generation.py", shell=True, capture_output=True, text=True)
print(P)
P = subprocess.run("python low_tf_model.py", shell=True, capture_output=True, text=True)
print(P)
P = subprocess.run("python high_tf_model.py", shell=True, capture_output=True, text=True)
print(P)
P = subprocess.run("python selection_model.py", shell=True, capture_output=True, text=True)
print(P)