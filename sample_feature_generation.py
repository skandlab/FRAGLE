import subprocess
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pysam
import os

input_bam = sys.argv[1]
output_path = sys.argv[2]
CPU = int(sys.argv[3])
bin_location_file = sys.argv[4]

bin_locations = pd.read_csv(bin_location_file).values.tolist()

final_hist = np.zeros(350)


def make_hist(sub_bin_locations):
    global input_bam
    hist = np.zeros(350)
    pybam = pysam.AlignmentFile(input_bam, "rb")
    command = f"samtools view -H {input_bam} | grep @SQ | grep chr"
    p = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
    flag = len(p.stdout)

    for bin_location in sub_bin_locations:
        contig, start, end = bin_location[0], int(bin_location[1]), int(bin_location[2])
        if flag == 0:
            contig = contig[3:]
        for read in pybam.fetch(contig, start, end):  # window
            if (
                (
                    (read.flag & 1024 == 0)
                    and (read.flag & 2048 == 0)
                    and (read.flag & 4 == 0)
                    and (read.flag & 8 == 0)
                )
                and read.mapping_quality >= 30
                and (read.reference_name == read.next_reference_name)
            ):
                length = read.template_length
                if length >= 51 and length <= 400:
                    hist[length - 51] += 1
    return hist


p = Pool(processes=CPU)
sub_bin_locations = np.array_split(bin_locations, CPU)
hist_list = p.map(make_hist, sub_bin_locations, 1)
p.close()
p.join()

for hist in hist_list:
    final_hist += hist
np.save(output_path, final_hist)
