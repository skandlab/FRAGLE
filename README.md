# FRAGLE
FRAGLE uses cfDNA fragment length features to prediction ctDNA fraction from a given sample.
If you have 100 samples and you want to get tumour fraction prediction for these 100 files, then do the following:
- Installation: You can directly install all required packages using the provided "requirements.txt" file. You can use Conda for this.
- Input:
    - For each sample, provide the (sample.bam, sample.bai) pair in a folder named according to your sample name/ID
    - Put that folder inside "bam_files" folder
    - Note that "sample.bam" and "sample.bai" should be the names used inside each of your your sample name/ID folder
    - See example here: https://bit.ly/3TpJk4N
- Running:
    - From the "Fragle" directory, run the "main.py" file
    - The following files will automatically run sequentially one after the other:
        - data_generation.py: generates low_data.pkl and high_data.pkl file inside "Intermediate_Files" folder using the raw bam files; will take some time to run
        - low_tf_model.py: takes the low_data.pkl file as input and outputs low_predictions.csv file inside "Intermediate_Files" folder 
        - high_tf_model.py: takes the high_data.pkl file as input and outputs high_predictions.csv file inside "Intermediate_Files" folder
        - selection_model.py: takes low_predictions.csv and high_predictions.csv file as input and outputs "Predictions.csv" file in the "Output" folder
- Output:
    - See the "Predictions.csv" file in the "Output" folder for output
    - "Predictions.csv" will be a two column (sample ID, predicted tumour fraction) csv file
    - You can see a sample output file inside the Output folder
- Other Relevant Info:
    - The three models used in Fragle are saved inside "models" folder
    - You can see the .ipynb version of the .py files inside "ipynb_version_code_files" folder (better readablity and documentation)
    - "meta_info" folder contains auxiliary files used for feature extraction, data normalization and model prediction
