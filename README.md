# Fragle
Fragle is an easy to use machine learning based software that can detect and quantify tumor fraction for all cancer types. It can work on both low pass whole genome sequencing (as low as ~0.05X) and targeted sequencing cell free DNA data derived from blood.<br>
![Fragle Overview](Fragle_Overview.png)

## Installation
- **Requirement**: You need to have `conda` available in your system.
- Download `Fragle.tar.gz` file from [Zenodo](https://zenodo.org/records/13968359).
- Command for using `Fragle.tar.gz` file for installation (see [Conda Pack](https://conda.github.io/conda-pack/) for more details):
  - `mkdir -p ./Fragle` (creating directory for unpacking)
  - `tar -xzf Fragle.tar.gz -C ./Fragle` (unpack the archive)
  - `source ./Fragle/bin/activate` (activate the environment)
    - Now, `Fragle/bin` is added to your path.
    - You can now run Fragle software.
  - `source ./Fragle/bin/deactivate` (deactivate environment)
- If you fail to install the software using the `Fragle.tar.gz` file, then you can manually install the following software libraries instead:
  - `Python 3.7+`, `Sklearn 1.2.2`, `Pandas`, `PyTorch` (CPU version), `Numpy`, `Pysam`, `SAMtools`
  - Use `pip` or `conda` to install the above-mentioned software libraries
  - You can use the PyTorch GPU version as well; make sure to install the appropriate GPU driver and CUDA version (see [PyTorch documentation](https://pytorch.org/))


## Fragle Input and Output Overview
- **Input**:
  - A directory full of WGS/Off-Target/Targeted Sequencing BAM files:
    - All BAM files inside the input directory must be of one type (either WGS or off-target or targeted sequencing).
    - All BAM files must be mapped to a specific reference genome (hg19 / GRCh37 / hg38).
    - The index file (.bai) must be provided with each BAM file.
  - In case the input directory contains targeted sequencing BAM files, you will also need to provide the BED file containing on-target sequenced regions.
- **Output**:
  - Fragle predicted ctDNA fractions for all BAM files in the input directory.
  - Processed feature file (.pkl) used by the Fragle models.
  - Off-target BAM files (if targeted sequencing BAM files are provided).
- **Example**:
  - Example BAM files are provided inside the `Input/` folder.
  - Outputs for the example BAM files can be found inside the `Output/` folder.
  - You can run the software on the example BAM files in the `Input/` folder and verify that the outputs match those in the `Output/` folder.


## Running Fragle
- Command:
    ```bash
    python main.py --input_folder <INPUT_FOLDER> --output_folder <OUTPUT_FOLDER> --option <OPTION> --bin_locations <BIN_LOCATIONS> --target_bed <TARGET_BED> --CPU <CPU> --threads <THREADS>
    ```
- Command Line Argument Description:
    - **INPUT_FOLDER**: Input folder path full of bam files and corresponding bai files (required string type argument)
    - **OUTPUT_FOLDER**: Output folder path where the Fragle predictions, processed features, and off-target bam files (in case of targeted sequencing bam files) will be found (required string type argument)
    - **OPTION**: 3 options: (1) 'F': run Fragle on processed features, (2) 'R': run Fragle on raw WGS/off-target bam files, (3) 'T': run Fragle on targeted sequencing bam files (required string type argument)
        - If you want to run Fragle directly on the processed features obtained from raw bam files (e.g., `Output/data.pkl`), you should use 'F' as the OPTION.
        - If you want to run Fragle on raw WGS or off-target bam files, use 'R' as the OPTION.
        - If you want to run Fragle on targeted sequencing (on+off target data) bam files, use 'T' as the OPTION.
    - **BIN_LOCATIONS**: CSV file containing genomic bin locations for Fragle feature generation from raw bam files (optional string type argument, default: `meta_info/hg19_bin_locations.csv`)
        - Choose `meta_info/hg19_bin_locations.csv` (default option) if your bam is mapped to hg19 or GRCh37 reference genome.
        - Choose `meta_info/hg38_bin_locations.csv` if your bam is mapped to hg38 reference genome.
    - **TARGET_BED**: bed file path for targeted sequencing bam file (optional string type argument, default: `empty.bed`)
        - This argument is only used when 'T' option is provided, meaning that you are running Fragle on targeted sequencing data.
        - The bed file is used to derive the off-target bam files from targeted bam files by Fragle (the off-target bam files can be found inside the OUTPUT_FOLDER).
    - **CPU**: Number of CPUs to use for parallel processing of bam files (integer type optional argument, default: 32)
        - If your running environment has multiple processors, setting the CPU to a higher value (e.g., 16 or 32) is recommended.
        - A higher CPU value will significantly speed up the software execution.
    - **THREADS**: Number of threads to use for off-target bam file extraction from targeted bam files (integer type optional argument, default: 32)
        - This argument is only utilized when the 'T' option is provided, meaning that you are running Fragle on targeted sequencing data.
        - A higher THREADS value will make the off-target bam extraction process significantly faster.


## Example Running Commands for Fragle
- **Running Fragle on hg19 mapped WGS BAM files:**
    ```bash
    python main.py --input_folder Input/ --output_folder Output/ --option R
    ```

- **Running Fragle on hg38 mapped WGS BAM files:**
    ```bash
    python main.py --input_folder Input/ --output_folder Output/ --option R --bin_locations meta_info/hg38_bin_locations.csv
    ```

- **Running Fragle on hg38 mapped off-target BAM files:**
    ```bash
    python main.py --input_folder Input/ --output_folder Output/ --option R --bin_locations meta_info/hg38_bin_locations.csv
    ```

- **Running Fragle on GRCh37 mapped targeted sequencing BAM files:**
    ```bash
    python main.py --input_folder Input/ --output_folder Output/ --option T --target_bed on_target.bed
    ```

- **Running Fragle on hg38 mapped targeted sequencing BAM files utilizing 16 threads and 16 CPUs:**
    ```bash
    python main.py --input_folder Input/ --output_folder Output/ --option T --bin_locations meta_info/hg38_bin_locations.csv --target_bed on_target.bed --CPU 16 --threads 16
    ```

- **Running Fragle on processed feature file (`data.pkl`) located inside `Input/` folder:**
    ```bash
    python main.py --input_folder Input/ --output_folder Output/ --option F
    ```


## Additional Information
- You will see the following files created in the specified output folder after running Fragle:
    - **data.pkl**: Feature file created in 'R' and in 'T' option → used by the models for ctDNA burden prediction
    - **HT.csv**: ctDNA predictions obtained from Fragle high ctDNA burden model
    - **LT.csv**: ctDNA predictions obtained from Fragle low ctDNA burden model
    - **Fragle.csv**: Final ctDNA predictions generated by the Fragle software
    - **off_target_bams/**:
        - This folder is created only when 'T' option is used
        - It contains off-target BAM files and corresponding index files extracted from the targeted BAM files
- If the coverage of any BAM file is too low (less than 0.025X), the software will generate a warning message for that BAM file besides generating the ctDNA fraction.


## Runtime and Memory Requirements:
- **20 MB of CPU memory** is sufficient to run Fragle.
- This CPU memory requirement remains constant even if you vary:
    - Number of BAM files in your input directory
    - Sequencing depth of the BAM files in your input directory
- **No GPU** is required
- It takes around **50 seconds** for Fragle to predict ctDNA fraction from a 1X WGS BAM utilizing only 1 CPU. The runtime can take only around **3 seconds** if you use the default CPU number of 32.
- The runtime increases **linearly** with the **increase of sequencing depth**


## Contacts
If you have any questions or feedback, please contact us at:<br>
rafeed.rahman015@gmail.com<br>
skanderupamj@gis.a-star.edu.sg