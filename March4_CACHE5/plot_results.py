import pandas as pd
from sklearn.metrics import confusion_matrix , f1_score

from Do_ML import ecpf4_featurizer
from March4_CACHE5.evaluation_functions import compute_thresholds , display_optimal_stats
from xgboost import XGBClassifier , XGBRegressor
#   Load XGBoost model
from xgboost import XGBRegressor , XGBClassifier
import pickle

import pandas as pd
from xgboost import XGBRegressor
import joblib

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def plot_pAct_distribution(data_frame, title= 'Distribution of pAct Values of Test Set'):
    plt.figure(figsize=(10, 6))
    # Check if Mean_pAct column exists and is not empty
    if 'pAct' in data_frame.columns and not data_frame['pAct'].isnull().all():
        plt.hist(data_frame['pAct'].dropna(), bins=30, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel('Mean_pAct')
        plt.ylabel('Frequency')
    else:
        print("Mean_pAct column is missing or contains no valid data.")
    plt.grid(True)
    plt.show()

pm = { 'Project_name' : 'March4_CACHE5' ,
       'dir' : '/home/weiser/PYTHON/CACHE5/' ,
       'data_dir' : '/Clusters_Max_TC' ,
       'data_file' : '/train_set_March4_CACHE5_',
       "model_dir" : '/models' ,
       'fig_dir' : '/figs' ,
       'num_test_set_clusters' : 50 ,  # number of clusters to make test set from
       'test_set_cluster_size' : 10 ,  # number of molecules to take from each cluster for test set
       'use_some_data' : 0 ,  # Use only 100 molecules for testing
       'maxevals' : 300 ,  # Number of evaluations for hyperopt
       'output_filename' : '/home/weiser/PYTHON/CACHE5/results.txt' ,  # File to write the results to
       'tan' : [ 0.3, 0.4 , 0.6 , 0.8 ] ,
       'hyp_tune' : 1,
       'regr' : 1,
        'class' : 0,
       'results' : 0,
       }


if pm['regr'] == 1:


   # For loading scaler objects

    # Define model and scaler paths
    model_scaler_paths = [
        { 'model' : '/home/weiser/PYTHON/CACHE5/March4_CACHE5/models/XGB_0.8_model.json' ,
          'scaler' : '/home/weiser/PYTHON/CACHE5/March4_CACHE5/models/0.8_scaler.pkl' } ,
        { 'model' : '/home/weiser/PYTHON/CACHE5/March4_CACHE5/models/XGB_0.6_model.json' ,
          'scaler' : '/home/weiser/PYTHON/CACHE5/March4_CACHE5/models/0.6_scaler.pkl' } ,
        { 'model' : '/home/weiser/PYTHON/CACHE5/March4_CACHE5/models/XGB_0.4_model.json' ,
          'scaler' : '/home/weiser/PYTHON/CACHE5/March4_CACHE5/models/0.4_scaler.pkl' }
    ]

    # Load models and scalers
    models_and_scalers = [ ]
    for entry in model_scaler_paths :
        model = XGBRegressor()
        model.load_model(entry[ 'model' ])  # Load the model
        scaler = joblib.load(entry[ 'scaler' ])  # Load the scaler
        models_and_scalers.append((model , scaler))

    # Load and prepare the test set
    # Assuming 'pm' and 'ecpf4_featurizer' are defined elsewhere in your code
    test_set = pd.read_csv(pm[ 'dir' ] + pm[ 'data_dir' ] + '/test_setMarch4_CACHE5.csv')
    x_test = test_set.drop('pAct' , axis=1)
    x_test = ecpf4_featurizer(x_test)
    y_test = test_set['pAct']

    # Prediction and Voting Mechanism

    binary_predictions = [ ]
    plot_pAct_distribution(test_set, 'Distribution of pAct Values of predicitions on Test Set')


    # Assuming x_test, actual_labels, and models_and_scalers are already defined
    activity_threshold = 7.3
    wiggle_rooms = np.linspace(-0.5 , 0.5 , 21)  # Define wiggle room values

    percent_active_correct = [ ]
    percent_inactive_discarded = [ ]

    # Get averaged prediction scores for each compound
    avg_predictions = np.zeros(len(x_test))
    for model , scaler in models_and_scalers :
        # Scale the test data with the corresponding scaler
        x_test_scaled = scaler.transform(x_test)

        # Predict using the scaled test data and accumulate predictions
        avg_predictions += model.predict(x_test_scaled)

    # Average the predictions
    avg_predictions /= len(models_and_scalers)

    #give avg_predictions_plot column of pAct
    avg_predictions_plot = pd.DataFrame(avg_predictions, columns=['pAct'])

    plot_pAct_distribution(avg_predictions_plot)

    for wiggle_room in wiggle_rooms :
        adjusted_threshold = activity_threshold - wiggle_room

        # Classify compounds based on the adjusted threshold
        predicted_active = avg_predictions >= adjusted_threshold

        # Calculate true positives and true negatives
        true_positives = np.sum(predicted_active & (y_test >= activity_threshold))
        true_negatives = np.sum(~predicted_active & (y_test < activity_threshold))

        # Calculate percentages
        percent_active = (true_positives / np.sum(y_test >= activity_threshold)) * 100
        percent_inactive = (true_negatives / np.sum(y_test < activity_threshold)) * 100

        # Append results for plotting
        percent_active_correct.append(percent_active)
        percent_inactive_discarded.append(percent_inactive)

    # Plotting the results
    plt.figure(figsize=(10 , 6))
    plt.plot(wiggle_rooms , percent_active_correct , label='% Active Correctly Predicted' , marker='o')
    plt.plot(wiggle_rooms , percent_inactive_discarded , label='% Inactive Correctly Discarded' , marker='x')
    plt.title('Impact of Wiggle Room on Model Predictions')
    plt.xlabel('Wiggle Room')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid(True)
    plt.show()

# inactive_labels now contains a 1 for compounds all models labeled as too inactive and 0 otherwise
if pm['class'] == 1:

    model_scaler_paths = [
        { 'model' : '/home/weiser/PYTHON/CACHE5/March4_CACHE5/models/XGBc_0.8_model.json' ,
          'scaler' : '/home/weiser/PYTHON/CACHE5/March4_CACHE5/models/0.8_scaler.pkl' } ,
        #{ 'model' : '/home/weiser/PYTHON/CACHE5/March4_CACHE5/models/XGBc_0.6_model.json' ,
        #  'scaler' : '/home/weiser/PYTHON/CACHE5/March4_CACHE5/models/0.6_scaler.pkl' } ,
        #{ 'model' : '/home/weiser/PYTHON/CACHE5/March4_CACHE5/models/XGBc_0.4_model.json' ,
        #  'scaler' : '/home/weiser/PYTHON/CACHE5/March4_CACHE5/models/0.4_scaler.pkl' }
    ]
    #TODO make this work with voting
    # Load models and scalers
    models_and_scalers = [ ]
    for entry in model_scaler_paths :
        model = XGBClassifier()
        model.load_model(entry[ 'model' ])  # Load the model
        scaler = joblib.load(entry[ 'scaler' ])  # Load the scaler
        models_and_scalers.append((model , scaler))

    # Load and prepare the test set
    # Assuming 'pm' and 'ecpf4_featurizer' are defined elsewhere in your code
    test_set = pd.read_csv(pm[ 'dir' ] + pm[ 'data_dir' ] + '/test_setMarch4_CACHE5.csv')
    x_test = test_set.drop('pAct' , axis=1)
    x_test = ecpf4_featurizer(x_test)
    y_test = test_set[ 'pAct' ]
    y_test_class = pd.Series(0 , index=y_test.index)  # Initialize with zeros
    y_test_class[ y_test > 8 ] = 1

    # Prediction
    # Get averaged prediction scores for each compound
    avg_predictions = np.zeros(len(x_test))
    for model , scaler in models_and_scalers :
        # Scale the test data with the corresponding scaler
        x_test_scaled = scaler.transform(x_test)

        # Predict using the scaled test data and accumulate predictions
        avg_predictions = model.predict_proba(x_test_scaled)

    # make avg_predictions 0 or 1
    avg_predictions = (avg_predictions > 0.5).astype(int)

    thresholds , threshold_95 , f1_scores = compute_thresholds(avg_predictions , y_test_class)
    optimal_threshold , accuracy , active_pos_rate , inactive_neg_rate , threshold_95 , accuracy_95 , active_acc_95 , inactive_acc_95  = display_optimal_stats(avg_predictions ,
                                                                                                          y_test_class ,
                                                                                                          thresholds ,
                                                                                                          threshold_95 ,
                                                                                                          f1_scores ,
                                                                                                            plot=True)





if pm['results']:
    # load results_all.csv
    data = pd.read_csv('/home/weiser/PYTHON/CACHE5/March4_CACHE5/results_all.csv')
    # First, let's read the uploaded CSV file to understand its structure and contents.

    # Extracting similarity values (assuming they are the max train-test similarity)
    similarity_values = data.columns[ 1 : ].astype(float)  # Convert column names to float for plotting

    # Extracting MAE and RMSE values
    mae_values = data.loc[ data[ 'Unnamed: 0' ] == 'Test MAE' ].iloc[ 0 , 1 : ].astype(float)
    rmse_values = data.loc[ data[ 'Unnamed: 0' ] == 'Test RMSE' ].iloc[ 0 , 1 : ].astype(float)

    # Plotting
    plt.figure(figsize=(10 , 6))
    plt.plot(similarity_values , mae_values , marker='o' , linestyle='-' , label='MAE')
    plt.plot(similarity_values , rmse_values , marker='s' , linestyle='--' , label='RMSE')

    plt.title('Test MAE and RMSE over Max Train-Test Similarity')
    plt.xlabel('Max Train-Test Similarity')
    plt.ylabel('Metric Value on Test Set')
    plt.legend()
    plt.grid(True)

    plt.show()





