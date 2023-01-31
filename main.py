import sys
import subprocess

input_path = sys.argv[1] 
output_folder = sys.argv[2]
if input_path.endswith('/')==False:
    input_path = input_path + '/'
if output_folder.endswith('/')==False:
    output_folder = output_folder + '/'

command = 'python data_generation.py {output_folder} {input_path}'.format(output_folder=output_folder, input_path=input_path)
subprocess.run(command, shell=True)

command = 'python LT_model.py {output_folder}'.format(output_folder=output_folder)
subprocess.run(command, shell=True)

command = 'python HT_model.py {output_folder}'.format(output_folder=output_folder)
subprocess.run(command, shell=True)

command = 'python select_model.py {output_folder}'.format(output_folder=output_folder)
subprocess.run(command, shell=True)