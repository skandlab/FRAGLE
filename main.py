import os
import argparse
import subprocess
import pathlib


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, nargs="*", required=True, help='Path(s) to either a single input folder with bam files, to one or more `.bam` files, or to a single data.pkl file with processed features.')
parser.add_argument('--output', type=str, required=True, help='Output folder path where the Fragle predictions and processed features can be found')
parser.add_argument('--mode', type=str, required=True, choices=["R", "T", "F"], help='3 options: (1) F -> output from processed feature, (2) R -> output from raw WGS/off-target bam file, (3) T -> output from targeted sequencing bam file')
parser.add_argument('--genome_build', type=str, default='hg19', help='reference genome version (hg19/GRCh37/hg38) to which your input bam files have been mapped to')
parser.add_argument('--target_bed', type=str, default='empty.bed', help='bed file for targeted sequencing (only utilized when T option is used)')
parser.add_argument('--cpu', type=int, default=32, help='Number of CPUs to use for parallel processing of bam files')
parser.add_argument('--threads', type=int, default=32, help='Number of threads to use for off-target bam file extraction from targeted bam files (only utilized when T option is used)')
args = parser.parse_args()

# Extract bam file paths or the data.pkl path

input_bam_paths = None
input_pkl_path = None
if len(args.input) == 1 and args.input[0][-4:].lower() not in [".bam", ".pkl"]:
    # Input folder specified
    input_folder = pathlib.Path(args.input[0])
    input_bam_paths = [path.resolve() for path in input_folder.glob("*.bam")]
    if len(input_bam_paths) == 0:
        raise RuntimeError(
            f"--input: No `.bam` files were found in the input folder: {input_folder}"
        )

elif len(args.input) == 1 and args.input[0][-4:].lower() == ".pkl":
    # Already processed features
    input_pkl_path = pathlib.Path(args.input[0])

else:
    # Direct paths to bam files
    if any([path[-4:].lower() != ".bam" for path in args.input]):
        raise ValueError(
            "--input: When specifying one or more `.bam` files, "
            "all paths in --input must end with '.bam'."
        )
    input_bam_paths = [pathlib.Path(path).resolve() for path in args.input]

# Check that we have the right input paths for the right options
if args.mode in ["R", "T"] and input_bam_paths is None:
    raise ValueError(
        "When --mode is 'R' or 'T', --input must specify the location(s) of `.bed` files."
    )
if args.mode == "F" and input_pkl_path is None:
    raise ValueError(
        "When --mode is 'F', --input must specify the location of the `data.pkl` file with processed features."
    )

output_folder = args.output
if output_folder[-1] != "/":
    output_folder += "/"

bin_locations = "meta_info/hg19_bin_locations.csv"
if args.genome_build == "hg38":
    bin_locations = "meta_info/hg38_bin_locations.csv"

# Create off-target bam files
if args.mode == "T":
    off_target_folder = output_folder + "off_target_bams/"
    if not os.path.isdir(off_target_folder):
        os.mkdir(off_target_folder)
    off_target_files = []
    for file_ in input_bam_paths:
        output_file = off_target_folder + file_.name
        command = f"samtools view -b -h -@ {args.threads} -o /dev/null -U {output_file} -L {args.target_bed} {file_}"
        subprocess.run(command, shell=True)
        command = f"samtools index -@ {args.threads} -b {output_file}"
        subprocess.run(command, shell=True)
        off_target_files += output_file

    # Use these for feature generation and prediction
    input_bam_paths = off_target_files

# Extract features from the bam files
if args.mode in ["R", "T"]:
    # Concatenate bam paths to a single string
    input_bam_paths_str = " ".join([str(path) for path in input_bam_paths])
    command = f"python feature_generation.py --input {input_bam_paths_str} --output {output_folder} --cpu {args.cpu} --bin_locations {bin_locations}"
    subprocess.run(command, shell=True)

    # Use these features (located in the output_folder) for prediction
    input_pkl_path = pathlib.Path(output_folder) / "data.pkl"

# Run the prediction
command = f"python predict.py --input {input_pkl_path} --output {output_folder}"  # data.pkl file inside output_folder folder
subprocess.run(command, shell=True)
