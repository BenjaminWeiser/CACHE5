import copy
import os

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import random

global id_in_test_train
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from tqdm import tqdm
import warnings

print(r"""
________________________________________________________________________________
 ______              _     _                                                        
|  ___ \            | |   (_)                                                       
| | _ | | ____  ____| | _  _ ____   ____                                            
| || || |/ _  |/ ___) || \| |  _ \ / _  )                                           
| || || ( ( | ( (___| | | | | | | ( (/ /                                            
|_||_||_|\_||_|\____)_| |_|_|_| |_|\____)                                                                            
 _                             _                                                    
| |                           (_)                                                   
| |      ____ ____  ____ ____  _ ____   ____                                        
| |     / _  ) _  |/ ___)  _ \| |  _ \ / _  |                                       
| |____( (/ ( ( | | |   | | | | | | | ( ( | |                                       
|_______)____)_||_|_|   |_| |_|_|_| |_|\_|| |                                       
                                      (_____|      _                                                     
   /\                                 _           | |                               
  /  \  _   _  ____ ____   ____ ____ | |_  ____ _ | |                               
 / /\ \| | | |/ _  |    \ / _  )  _ \|  _)/ _  ) || |                               
| |__| | |_| ( ( | | | | ( (/ /| | | | |_( (/ ( (_| |                               
|______|\____|\_|| |_|_|_|\____)_| |_|\___)____)____|                               
 _____       (_____|_     _                                                                                    
(____ \            | |   (_)                                                        
 _   \ \ ___   ____| |  _ _ ____   ____                                             
| |   | / _ \ / ___) | / ) |  _ \ / _  |                                            
| |__/ / |_| ( (___| |< (| | | | ( ( | |                                            
|_____/ \___/ \____)_| \_)_|_| |_|\_|| |                                            
                                 (_____|                                            



  __       ______ _     _ ______     _____       _     _ _     _      _             
 /  |     / _____) |   | (_____ \   (_____)     | |   (_) |   (_)_   (_)            
/_/ |    | /     | |___| |_____) )     _   ____ | | _  _| | _  _| |_  _  ___  ____  
  | |    | |      \_____/|  ____/     | | |  _ \| || \| | || \| |  _)| |/ _ \|  _ \ 
  | |_   | \_____   ___  | |         _| |_| | | | | | | | |_) ) | |__| | |_| | | | |
  |_(_)   \______) (___) |_|        (_____)_| |_|_| |_|_|____/|_|\___)_|\___/|_| |_|

____________________________________________________________________________________
____________________________________________________________________________________

                      Author: Benjamin Kachkowski Weiser
                        Date: September 6, 2023

This script embodies cutting-edge algorithms to augment traditional molecular docking, FITTED, processes
with machine learning for the prediction of Cytochrome P450 (CYP) enzyme inhibition.

Sections:

    1. Cleaned and combined test and train Pei sets with CYP_clean_files.ipnyb. Sets combined and then clustered to create new train and test sets
    2. Dock each ligand 5 times to its respective isoform using FITTED. Docked data can be found here: (to be inserted)
    3. Create analogue sets using FITTED. Create max train-test similarity using CYP_TC_DataSets.py
    4. Run RF with Feature Importances using max train-test similarity of 0.8 using ML_over_Tanimoto.py which calls CYP_inhibition_functions.py and Do_ML2.py
    5. Using these selected features run all ML models on all datasets using ML_over_Tanimoto.py which calls CYP_inhibition_functions.py and Do_ML2.py
    6. Use CYP_evaluate_and_ensemble.py which calls CYP_evaluate_and_ensemble_functions.py to make ensembles and evaluate and graph model performance

Please ensure you have all the required libraries installed.
For any issues or questions, please contact: benjamin.weiser@mail.mcgill.ca
Github: https://github.com/MoitessierLab/ML-augmented-docking-CYP-inhibition

____________________________________________________________________________________
____________________________________________________________________________________

""")


# Set seed
seed = 1
os.environ['PYTHONHASHEDSEED'] = str(seed)
np.random.seed(seed)

'''Here's a high level overview of what this script does:
load_parameters(): This function returns a dictionary of parameters required for the operation of the script.
Next, a directory is created if it doesn't already exist.

Loop over different Cytochrome P450 isoforms. For each isoform:

a. Read in SMILES (Simplified Molecular Input Line Entry System) strings for inactive and active states of the isoform.
b. Depending on the parameters specified, it reads and filters molecules that have been docked / scored. It ensures all the SMILE strings used have been docked before.
c. If needed, it can also limit the dataset to a specific smaller number of molecules, or adjust the datasets based on the clusters.
d. The script then constructs a test set of molecules based on specified conditions (like Tanimoto coefficient).
e. The script creates a training set, ensuring none of the samples are in the test set, and the most similar one to the test set has a Tanimoto similarity below a certain threshold. Different training sets are created for various Tanimoto thresholds.

It saves detailed information about the training sets and Tanimoto coefficients to a CSV file.
Finally, it plots the size of the train set against different Tanimoto coefficients and saves this plot as a PNG file.
Overall this script is a mix of data preparation, filtering based on the application of certain conditions and measures (Tanimoto coefficient calculation) and basic exploratory data analysis (creation of plots). 
'''
def load_parameters():
    pm = {'Project_name': 'March4_CACHE5',
          'dir': '/home/weiser/PYTHON/CACHE5/',
          'num_test_set_clusters': 50, # number of clusters to make test set from
          'test_set_cluster_size': 10, # number of molecules to take from each cluster for test set
          'use_some_data': 0, # Use only 500 molecules for testing
          }
    return pm

tanimoto = [0.3, 0.4, 0.6, 0.8]
# make a dataframe with columns for each tanimoto value


pm = load_parameters()
DATA_DIR = pm['dir']

# make directory for tanimoto smiles csv in pm['dir']
if not os.path.exists(pm['dir'] + '/Clusters_Max_TC'):
    os.makedirs(pm['dir'] + '/Clusters_Max_TC')

#load data from data_dir + cahce5_data.csv
data = pd.read_csv(DATA_DIR + 'cache5_data.csv')
#drop duplicate smiles
data = data.drop_duplicates(subset=['SMILES'])

if pm['use_some_data'] == 1:
    data = data.head(500)

tanimoto_df = pd.DataFrame(columns=tanimoto)
# Make Test Set
##############################################################################################################
print('Making Test Set')
test_set = pd.DataFrame(columns=['SMILES', 'pAct'])
for i in tqdm(range(pm['num_test_set_clusters']), total=pm['num_test_set_clusters'], desc="Making Test Set"):
    origin_seed = random.randint(0, len(data)) - 1
    test_set_origin = data['SMILES'].iloc[origin_seed]  
    # pick a random id from docked_ids to be the test set origin
    # Get the tanimoto similarity between the test set origin and all uniques docked smiles
    # keep these 500 and call them the test ids
    
    origin_mol = AllChem.MolFromSmiles(test_set_origin)
    origin_bit = AllChem.GetMorganFingerprintAsBitVect(origin_mol, radius=2, nBits=2048)
    origin_tanimoto_coefficient = pd.DataFrame(columns=['SMILES', 'tanimoto_coefficient'])
    for test_compound in data['SMILES']:
        test_mol = Chem.MolFromSmiles(test_compound)
        test_bit = AllChem.GetMorganFingerprintAsBitVect(test_mol, radius=2, nBits=2048)
        tanimoto_coefficient = DataStructs.TanimotoSimilarity(origin_bit, test_bit)
        # add tano to origin_tanimoto_coefficient and test_compound
        add = pd.DataFrame({'SMILES': [test_compound], 'tanimoto_coefficient': [tanimoto_coefficient]})
        if not add.empty and not add.isna().all().all():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                origin_tanimoto_coefficient = pd.concat([origin_tanimoto_coefficient, add])

    # add id column of data to origin_tanimoto_coefficient and merge on smiles
    test_set_cluster = origin_tanimoto_coefficient.merge(data, on='SMILES')
    # sort origin_tanimoto_coefficient by tanimoto_coefficient and take top 500
    test_set_cluster = test_set_cluster.sort_values(by=['tanimoto_coefficient'], ascending=False)
    test_set_cluster = test_set_cluster.head(pm['test_set_cluster_size'])
    test_set = pd.concat([test_set, test_set_cluster], ignore_index=True)

# get size for test set dataframe
print('Size of test set:', len(test_set))
test_set.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'test_set' + pm['Project_name'] + '.csv', index=False)

# Make Train Set
##############################################################################################################

print('Making Train Set')
# drop test set from data
train_smiles = copy.deepcopy(data)
# drop test set from train_smiles
train_smiles = train_smiles[~train_smiles['SMILES'].isin(test_set['SMILES'])]

most_similar_to_in_train = pd.DataFrame(columns=['SMILES', 'tanimoto_coefficient'])
for train_compound in tqdm(train_smiles['SMILES'], total=len(train_smiles), desc="Making Train Set"):
    train_mol = Chem.MolFromSmiles(train_compound)
    train_bit = AllChem.GetMorganFingerprintAsBitVect(train_mol, radius=2, nBits=2048)
    train_simularities = pd.DataFrame(columns=['tanimoto_coefficient'])
    for test_compound in test_set['SMILES']:
        # Convert the train compound to a mol object
        test_mol = Chem.MolFromSmiles(test_compound)
        test_bit = AllChem.GetMorganFingerprintAsBitVect(test_mol, radius=2, nBits=2048)
        tanimoto_coefficient = DataStructs.TanimotoSimilarity(test_bit, train_bit)
        new_row = pd.DataFrame({'tanimoto_coefficient': [tanimoto_coefficient]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_simularities = pd.concat([train_simularities, new_row], ignore_index=True)


    max_tanimoto = train_simularities['tanimoto_coefficient'].max()
    new_row = pd.DataFrame({'SMILES': [train_compound], 'tanimoto_coefficient': [max_tanimoto]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        most_similar_to_in_train = pd.concat([most_similar_to_in_train, new_row], ignore_index=True)

for MAX_TANIMOTO in tanimoto:
    # keep only rows of most_similar_to_in_train where tanimoto_coefficient is less than 0.5
    most_similar_to_in_train_tan = most_similar_to_in_train[
        most_similar_to_in_train['tanimoto_coefficient'] < MAX_TANIMOTO]
    train_set = most_similar_to_in_train_tan.merge(data, on='SMILES')
    # get size for train set dataframe
    print('TAN:', MAX_TANIMOTO, 'Size train:', len(train_set))
    # put size of train_set into row 'train size and tanimoto column into tanimoto_df for plotting
    tanimoto_df.loc['train_set_size', MAX_TANIMOTO] = len(train_set)
    train_set.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'train_set_' + pm['Project_name'] + '_' + str(MAX_TANIMOTO) + '.csv',
                     index=False)
    tanimoto_df.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'tanimoto_df_' + pm['Project_name'] + '.csv', index=False)

# Plot using matplot lib train_set_size over tanimoto
plt.plot(tanimoto_df.columns, tanimoto_df.loc['train_set_size'])
plt.xlabel('Tanimoto Coefficient')
plt.ylabel('Train Set Size')
plt.title('Train Set Size vs Tanimoto Coefficient')
# add y-axis value to each point
for i in tanimoto:
    plt.annotate(tanimoto_df.loc['train_set_size'][i], (i, tanimoto_df.loc['train_set_size'][i]))
plt.savefig(pm['dir'] + '/Clusters_Max_TC/' + 'train_set_size_vs_tanimoto_' + pm['Project_name'] + '.png')
plt.close()


