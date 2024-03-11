import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import matplotlib.pyplot as plt

# get sdf file and prepare the data for the cache5 model from /home/weiser/Documents/CACHE5/MCHR1_ChEMBL.sdf and /home/weiser/Documents/CACHE5/MCHR1_patent.sdf

def read_sdf_and_extract_data(file_path, columns=None):
    sdf_data = PandasTools.LoadSDF(file_path, smilesName='SMILES', molColName='Molecule', includeFingerprints=False)
    # If specific columns are requested, extract those.
    if columns:
        sdf_data = sdf_data[columns]
    return sdf_data



def prepare_data_for_cache5(data_frame):
    # Ensure Mean_pAct is a float for accurate calculations and plotting
    if 'Mean_pAct' in data_frame.columns:
        data_frame['Mean_pAct'] = pd.to_numeric(data_frame['Mean_pAct'], errors='coerce')
    return data_frame


def plot_mean_pAct_distribution(data_frame):
    plt.figure(figsize=(10, 6))
    # Check if Mean_pAct column exists and is not empty
    if 'Mean_pAct' in data_frame.columns and not data_frame['Mean_pAct'].isnull().all():
        plt.hist(data_frame['Mean_pAct'].dropna(), bins=30, color='skyblue', edgecolor='black')
        plt.title('Distribution of Mean_pAct Values in CHEMBL Data')
        plt.xlabel('Mean_pAct')
        plt.ylabel('Frequency')
    else:
        print("Mean_pAct column is missing or contains no valid data.")
    plt.grid(True)
    plt.show()

def plot_pAct_distribution(data_frame):
    plt.figure(figsize=(10, 6))
    # Check if Mean_pAct column exists and is not empty
    if 'pAct' in data_frame.columns and not data_frame['pAct'].isnull().all():
        plt.hist(data_frame['pAct'].dropna(), bins=30, color='skyblue', edgecolor='black')
        plt.title('Distribution of pAct Values of CHEMBL and Patent Data')
        plt.xlabel('Mean_pAct')
        plt.ylabel('Frequency')
    else:
        print("Mean_pAct column is missing or contains no valid data.")
    plt.grid(True)
    plt.show()







from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd

from rdkit.Chem import PandasTools
import pandas as pd


from rdkit.Chem import PandasTools
import pandas as pd

from rdkit.Chem import PandasTools
import pandas as pd

def read_sdf_and_extract_activities(file_path):
    sdf_data = PandasTools.LoadSDF(file_path, smilesName='SMILES', molColName='Molecule', includeFingerprints=False)
    extracted_data = []

    for index, row in sdf_data.iterrows():
        smiles = row['SMILES']
        acnames = row.get('acname', '').split('\n')
        acvalues = row.get('acvalue_uM', '').split('\n')
        acvalues = [float(acvalue) for acvalue in acvalues if acvalue]

        # Initialize a dictionary to hold the activity data for Ki and IC50 separately
        activity_data = {'SMILES': smiles, 'Ki': None, 'IC50': None}

        for acname, acvalue in zip(acnames, acvalues):
            if acname == 'Ki' and activity_data['Ki'] is None:
                activity_data['Ki'] = acvalue
            elif acname == 'IC50' and activity_data['IC50'] is None:
                activity_data['IC50'] = acvalue

        # Only add to extracted_data if at least one of Ki or IC50 is present
        if activity_data['Ki'] is not None or activity_data['IC50'] is not None:
            extracted_data.append(activity_data)
        else:
            print(f"No Ki or IC50 data for SMILES {smiles}, discarding.")

    return pd.DataFrame(extracted_data)




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt

def plot_activities_distribution(data_frame, max_ki, max_ic50):
    """
    Plots histograms for the distribution of Ki and IC50 values.

    Parameters:
    - data_frame: DataFrame, the data containing Ki and IC50 values.
    """
    #delete is ki is greater than 1
    data_frame = data_frame[data_frame['Ki'] <= max_ki]
    data_frame = data_frame[data_frame['IC50'] <= max_ic50]

    # Set up the matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(30, 10), sharey=True)
    fig.suptitle('Distribution of Activity Values')


    # Plot for Ki values
    ki_values = data_frame['Ki'].dropna()
    axes[0].hist(ki_values, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('Distribution of Ki Values')
    axes[0].set_xlabel('Ki (uM)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True)


    # Plot for IC50 values
    ic50_values = data_frame['IC50'].dropna()
    axes[1].hist(ic50_values, bins=30, color='lightgreen', edgecolor='black')
    axes[1].set_title('Distribution of IC50 Values')
    axes[1].set_xlabel('IC50 (uM)')
    # No need for y-label here; it's shared with the first plot
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    plt.show()





# Paths to the SDF files (Update these paths as per your environment)
chembl_sdf_path = '/home/weiser/Documents/CACHE5/MCHR1_ChEMBL.sdf'
patent_sdf_path = '/home/weiser/Documents/CACHE5/MCHR1_patent.sdf'

# Read and prepare ChEMBL data
chembl_data = read_sdf_and_extract_data(chembl_sdf_path)
chembl_data_prepared = prepare_data_for_cache5(chembl_data)

# Plotting the distribution of mean_pAct values
plot_mean_pAct_distribution(chembl_data_prepared)


patent_data = read_sdf_and_extract_data(patent_sdf_path)
patent_data_prepared = read_sdf_and_extract_activities(patent_sdf_path)
# Assuming 'patent_data_processed' is your DataFrame after processing
plot_activities_distribution(patent_data_prepared, 1, 1)
plot_activities_distribution(patent_data_prepared, 0.025, 30)

#take patent_data_prepared and take only ic50 column. drop nan rows
patent_data_prepared = patent_data_prepared.dropna(subset=['SMILES','IC50']) #460 have ic50 data
# ic50 are in micro molar
#convert ic50 to pAct
patent_data_prepared['pAct'] = -1 * np.log10(patent_data_prepared['IC50'] * 10**-6)

#save smiles and pact of patent_data_prepared and chembl_data_prepared together in file call chache5_data.csv
patent_data_prepared = patent_data_prepared[['SMILES', 'pAct']]
chembl_data_prepared = chembl_data_prepared[['SMILES', 'Mean_pAct']]
#remname mean_pact to pact
chembl_data_prepared = chembl_data_prepared.rename(columns={'Mean_pAct': 'pAct'})
#combine patent_data_prepared and chembl_data_prepared
cache5_data = pd.concat([patent_data_prepared, chembl_data_prepared], ignore_index=True)
#save cache5_data to csv
cache5_data.to_csv('cache5_data.csv', index=False)
#plot the distribution of pAct values
plot_pAct_distribution(cache5_data)




