import argparse
import os
import subprocess
import shutil

# Set FRAGLE_HOME to the directory of main.py
script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["FRAGLE_HOME"] = script_dir

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Input folder path full of bam files')
parser.add_argument('--output', type=str, required=True, help='Output folder path where the Fragle predictions and processed features can be found')
parser.add_argument('--mode', type=str, required=True, help='3 options: (1) F -> output from processed feature, (2) R -> output from raw WGS/off-target bam file, (3) T -> output from targeted sequencing bam file')
parser.add_argument('--genome_build', type=str, default='hg19', help='reference genome version (hg19/GRCh37/hg38) to which your input bam files have been mapped to')
parser.add_argument('--target_bed', type=str, default='empty.bed', help='bed file for targeted sequencing (only utilized when T option is used)')
parser.add_argument('--cpu', type=int, default=32, help='Number of CPUs to use for parallel processing of bam files')
parser.add_argument('--threads', type=int, default=32, help='Number of threads to use for off-target bam file extraction from targeted bam files (only utilized when T option is used)')

args = parser.parse_args()
input_path = args.input
output_base = args.output
option = args.mode
genome_build = args.genome_build
bed_file = args.target_bed
CPU = args.cpu
num_threads = args.threads

# Handle input (file or directory)
file_flag = 0
copied_bam, copied_bai = '', ''
if os.path.isfile(input_path) and input_path.endswith(".bam"):
    # Create output subdirectory named after the BAM file
    bam_basename = os.path.splitext(os.path.basename(input_path))[0]
    sample_output_dir = os.path.join(output_base, bam_basename)
    os.makedirs(sample_output_dir, exist_ok=True)

    # Check for BAI file
    bai_path = f"{input_path}.bai"
    if not os.path.exists(bai_path):
        raise FileNotFoundError(f"Index file {bai_path} required in the form of [bam_name].bam.bai")

    # Copy BAM and BAI to the output subdirectory
    shutil.copy(input_path, sample_output_dir)
    copied_bam = os.path.join(sample_output_dir, f'{bam_basename}.bam')
    shutil.copy(bai_path, sample_output_dir)
    copied_bai = f'{copied_bam}.bai'
    file_flag=1

    input_folder = os.path.join(sample_output_dir, "")
    output_folder = os.path.join(sample_output_dir, "")
else:
    input_folder = os.path.join(input_path, "")
    output_folder = os.path.join(output_base, "")
    os.makedirs(output_folder, exist_ok=True)

# Resolve bin_locations using FRAGLE_HOME
bin_locations = os.path.join(
    os.environ["FRAGLE_HOME"], "meta_info", "hg19_bin_locations.csv"
)
if genome_build == "hg38":
    bin_locations = os.path.join(
        os.environ["FRAGLE_HOME"], "meta_info", "hg38_bin_locations.csv"
    )

if option == "R":
    command = f"python {os.path.join(os.environ['FRAGLE_HOME'], 'feature_generation.py')} {input_folder} {output_folder} {CPU} {bin_locations}"
    subprocess.run(command, shell=True, check=False)
    command = f"python {os.path.join(os.environ['FRAGLE_HOME'], 'predict.py')} {output_folder} {output_folder}"
    subprocess.run(command, shell=True, check=False)
elif option == "F":
    command = f"python {os.path.join(os.environ['FRAGLE_HOME'], 'predict.py')} {input_folder} {output_folder}"
    subprocess.run(command, shell=True, check=False)
elif option == "T":
    off_target_folder = os.path.join(output_folder, "off_target_bams")
    os.makedirs(off_target_folder, exist_ok=True)
    for file_ in os.listdir(input_folder):
        if file_.endswith(".bam"):
            input_file = os.path.join(input_folder, file_)
            output_file = os.path.join(off_target_folder, file_)
            command = f"samtools view -b -h -@ {num_threads} -o /dev/null -U {output_file} -L {bed_file} {input_file}"
            subprocess.run(command, shell=True, check=False)
            command = f"samtools index -@ {num_threads} -b {output_file}"
            subprocess.run(command, shell=True, check=False)
    input_folder = off_target_folder
    command = f"python {os.environ['FRAGLE_HOME']}/feature_generation.py {input_folder} {output_folder} {CPU} {bin_locations}"
    subprocess.run(command, shell=True, check=False)
    command = f"python {os.environ['FRAGLE_HOME']}/predict.py {output_folder} {output_folder}"
    subprocess.run(command, shell=True, check=False)

if file_flag==1:
    os.remove(copied_bam)
    os.remove(copied_bai)