import os, argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, required=True, help='Input folder path full of bam files')
parser.add_argument('--output_folder', type=str, required=True, help='Output folder path where the Fragle predictions and processed features can be found')
parser.add_argument('--option', type=str, required=True, help='3 options: (1) F -> output from processed feature, (2) R -> output from raw WGS/off-target bam file, (3) T -> output from targeted sequencing bam file')
parser.add_argument('--bin_locations', type=str, default='meta_info/hg19_bin_locations.csv', help='CSV file containing genomic bin locations for feature generation: Choose meta_info/hg19_bin_locations.csv (default option) if your bam is mapped to hg19 or GRCh37 reference genome; choose meta_info/hg38_bin_locations.csv if your bam is mapped to hg38 reference genome')
parser.add_argument('--target_bed', type=str, default='empty.bed', help='bed file for targeted sequencing (only utilized when T option is used)')
parser.add_argument('--CPU', type=int, default=32, help='Number of CPUs to use for parallel processing of bam files')
parser.add_argument('--threads', type=int, default=32, help='Number of threads to use for off-target bam file extraction from targeted bam files (only utilized when T option is used)')

args = parser.parse_args()
input_folder = args.input_folder + '/'
output_folder = args.output_folder + '/'
option = args.option
bin_locations = args.bin_locations
bed_file = args.target_bed
CPU = args.CPU
num_threads = args.threads


if option=='R':
    command = 'python feature_generation.py {input} {output} {CPU} {bin_locations}'.format(input=input_folder, output=output_folder, CPU=CPU, bin_locations=bin_locations)
    subprocess.run(command, shell=True)
    command = 'python predict.py {input} {output}'.format(input=output_folder, output=output_folder) # data.pkl file inside output_folder folder
    subprocess.run(command, shell=True)

elif option=='F':
    command = 'python predict.py {input} {output}'.format(input=input_folder, output=output_folder) # data.pkl file inside input_folder folder
    subprocess.run(command, shell=True)

elif option=='T':
    off_target_folder = output_folder + 'off_target_bams/'
    if os.path.isdir(off_target_folder)==False:
        os.mkdir(off_target_folder)
    for file_ in os.listdir(input_folder):
        if file_.endswith('.bam'):
            input_file = input_folder + file_
            output_file = off_target_folder + file_
            command = 'samtools view -b -h -@ {num_threads} -o /dev/null -U {off_bam} -L {bed_file} {in_bam}'.format(num_threads=num_threads, off_bam=output_file, bed_file=bed_file, in_bam=input_file)
            subprocess.run(command, shell=True)
            command = 'samtools index -@ {num_threads} -b {file_}'.format(num_threads=num_threads, file_=output_file)
            subprocess.run(command, shell=True)

    input_folder = off_target_folder
    command = 'python feature_generation.py {input} {output} {CPU} {bin_locations}'.format(input=input_folder, output=output_folder, CPU=CPU, bin_locations=bin_locations)
    subprocess.run(command, shell=True)
    command = 'python predict.py {input} {output}'.format(input=output_folder, output=output_folder) # data.pkl file inside output_folder folder
    subprocess.run(command, shell=True)
