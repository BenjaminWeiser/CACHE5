

from Do_ML import ecpf4_featurizer
from Prepare_CACHE5_data import plot_pAct_distribution
from evaluation_functions import compute_thresholds , display_optimal_stats
from xgboost import XGBClassifier , XGBRegressor
#   Load XGBoost model
from xgboost import XGBRegressor , XGBClassifier
import pickle

import pandas as pd
from xgboost import XGBRegressor
import joblib

import numpy as np
import matplotlib.pyplot as plt




pm = { 'Project_name' : 'March11_CACHE5_2' ,
       'dir' : 'CACHE5' ,
       'data_dir' : 'Clusters_Max_TC' ,
       'data_file' : '/train_set_March4_CACHE5_',
       "model_dir" : '/models' ,
       'fig_dir' : '/figs' ,
       'num_test_set_clusters' : 50 ,  # number of clusters to make test set from
       'test_set_cluster_size' : 10 ,  # number of molecules to take from each cluster for test set
       'use_some_data' : 0 ,  # Use only 100 molecules for testing
       'maxevals' : 300 ,  # Number of evaluations for hyperopt
       'output_filename' : '/results.txt' ,  # File to write the results to
       'tan' : [ 0.3, 0.4 , 0.6 , 0.8 ] ,
       'hyp_tune' : 1,
       'regr' : 0,
        'class' : 1,
       'results' : 0,
       }


if pm['regr'] == 1:


   # For loading scaler objects

    # Define model and scaler paths
    model_scaler_paths = [
        { 'model' : 'March4_CACHE5/models/XGB_0.8_model.json' ,
          'scaler' : 'March4_CACHE5/models/0.8_scaler.pkl' } ,
        { 'model' : 'March4_CACHE5/models/XGB_0.6_model.json' ,
          'scaler' : 'March4_CACHE5/models/0.6_scaler.pkl' } ,
        { 'model' : 'March4_CACHE5/models/XGB_0.4_model.json' ,
          'scaler' : 'March4_CACHE5/models/0.4_scaler.pkl' }
    ]
    import os

    '''file_path = 'models/XGB_0.8_model.json'
    abs_file_path = os.path.abspath(file_path)

    print(f"Absolute path: {abs_file_path}")

    if os.path.isfile(file_path):
        print("File exists")
    else:
        print("File does not exist")'''
    # Load models and scalers
    models_and_scalers = [ ]
    for entry in model_scaler_paths :
        model = XGBRegressor()
        model.load_model(entry[ 'model' ])  # Load the model
        scaler = joblib.load(entry[ 'scaler' ])  # Load the scaler
        models_and_scalers.append((model , scaler))

    # Load and prepare the test set
    # Assuming 'pm' and 'ecpf4_featurizer' are defined elsewhere in your code
    test_set = pd.read_csv(pm[ 'data_dir' ] + '/test_setMarch4_CACHE5.csv')
    x_test = test_set.drop('pAct' , axis=1)
    x_test = ecpf4_featurizer(x_test)
    y_test = test_set['pAct']

    # Prediction and Voting Mechanism

    binary_predictions = [ ]
    plot_pAct_distribution(test_set, 'Distribution of pAct Values of predicitions on Test Set')
    i = 0


    # Assuming x_test, actual_labels, and models_and_scalers are already defined
    activity_thresholds = [7.3, 7, 7.8, 8]
    for activity_threshold in activity_thresholds:
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

        if i ==0:
            plot_pAct_distribution(avg_predictions_plot)
            i =1

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
        plt.title('Impact of Wiggle Room on Model Predictions with threshold = ' + str(activity_threshold))
        plt.xlabel('Wiggle Room')
        plt.ylabel('Percentage')
        plt.legend()
        plt.grid(True)
        plt.show()

# inactive_labels now contains a 1 for compounds all models labeled as too inactive and 0 otherwise
if pm['class'] == 1:

    model_scaler_paths = [
        { 'model' : 'March4_CACHE5/models/XGBc_0.8_model.json' ,
          'scaler' : 'March4_CACHE5/models/0.8_scaler.pkl' } ,
        #{ 'model' : 'March4_CACHE5/models/XGBc_0.6_model.json' ,
        #  'scaler' : 'March4_CACHE5/models/0.6_scaler.pkl' } ,
        #{ 'model' : 'March4_CACHE5/models/XGBc_0.4_model.json' ,
        #  'scaler' : 'March4_CACHE5/models/0.4_scaler.pkl' }
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
    test_set = pd.read_csv(pm[ 'data_dir' ] + '/test_setMarch4_CACHE5.csv')
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
    data = pd.read_csv('March11_CACHE5_2/results_all.csv')
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





