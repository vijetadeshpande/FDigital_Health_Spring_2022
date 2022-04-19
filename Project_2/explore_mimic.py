import pandas as pd
import os


#
path_mimic = r'/Users/vijetadeshpande/Downloads/BioNLP Lab/Datasets/EHR/mimic-iii-clinical-database-1.4'

files = ['NOTEEVENTS', 'D_ICD_DIAGNOSES']
data_raw = {}
for file in files:
    data_raw[file] = pd.read_csv(os.path.join(path_mimic, file+'.csv'))

#
data_raw['NOTEEVENTS'].iloc[:1000, :].to_csv(os.path.join(path_mimic, 'NOTEEVENTS_sample.csv'))