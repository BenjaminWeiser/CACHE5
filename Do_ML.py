import gc
import os
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
from hyperopt import fmin , tpe , hp , STATUS_OK , Trials
from hyperopt.pyll import scope
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score , train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier , XGBRegressor


def Do_XGradientBoost_regression(X , y , testSet , scores_test , name , pm , scored , seed) :
    m = 'XGBr'
    scaler = MinMaxScaler()

    # Fit and transform X and transform testSet
    X = scaler.fit_transform(X)
    testSet_norm = scaler.transform(testSet)
    # Save normalizing parameters
    scaler_path = pm[ 'dir' ] + '/' + pm[ 'Project_name' ] + pm[ 'model_dir' ] + '/' + name + '_scaler.pkl'
    joblib.dump(scaler , scaler_path)

    print(
        "|---------------------------------------------------\nStarting Hyp Search for XGBoost Regressor\n|---------------------------------------------------")

    if pm[ 'hyp_tune' ] == 1 :
        X_tune , _ , y_tune , _ = train_test_split(X , y , test_size=0.2 , random_state=seed)

        def hyperopt_train_test(params) :
            modelxgb = XGBRegressor(**params , random_state=seed)
            return -cross_val_score(modelxgb , X_tune , y_tune , scoring='neg_mean_squared_error' , cv=5 ,
                                    n_jobs=-2).mean()

        space4xgb = { 'n_estimators' : scope.int(hp.quniform('n_estimators' , 5 , 100 , 1)) ,
                      'learning_rate' : hp.choice('learning_rate' , [ 0.015 , 0.03 , 0.06 , 0.12 , 0.25 ]) ,
                      'max_depth' : scope.int(hp.quniform('max_depth' , 10 , 20 , 1)) ,
                      'min_child_weight' : scope.int(hp.quniform('min_child_weight' , 1 , 10 , 1)) ,
                      'gamma' : scope.int(hp.quniform('gamma' , 0 , 9 , 1)) ,
                      'reg_alpha' : scope.int(hp.quniform('reg_alpha' , 0 , 180 , 1)) ,
                      'reg_lambda' : scope.int(hp.quniform('reg_lambda' , 0 , 100 , 1)) ,
                      'subsample' : hp.uniform('subsample' , 0.5 , 1.0) ,
                      'colsample_bytree' : hp.uniform('colsample_bytree' , 0.5 , 1.0) ,
                      'colsample_bylevel' : hp.uniform('colsample_bylevel' , 0.5 , 1.0) ,
                      'colsample_bynode' : hp.uniform('colsample_bynode' , 0.5 , 1.0)
                      }

        best_score = 10
        best_params = None

        def f(params) :
            nonlocal best_score
            nonlocal best_params
            acc = hyperopt_train_test(params)
            # print('acc:', acc)
            if acc < best_score :
                best_score = acc
                best_params = params
                print('new best:' , best_score , params)
            return { 'loss' : acc , 'status' : STATUS_OK }

        trials = Trials()
        best = fmin(f , space4xgb , algo=tpe.suggest , max_evals=pm[ 'maxevals' ] , trials=trials)

        # Print the best parameters
        print("Best parameters:" , best_params)

        log_best_params(name , best_params , pm)

        plot_hyperopt_results(best_params , trials , pm , 'XGB' , name)

    model = XGBRegressor(**best_params , random_state=seed)
    model.fit(X , y, scoring= 'mean_squared_error')

    scores_pred_test = model.predict(testSet_norm)

    eval = evaluate_regression_model(y , model.predict(X) , model , testSet_norm , scores_test)
    print(eval)

    save_model_xgb(model , pm , 'XGB' , name)
    score_xgb = pd.DataFrame.from_dict(eval , orient='index' , columns=[ name ])
    return score_xgb


def log_best_params(name , best , pm) :
    with open(pm[ 'output_filename' ] , 'a') as fileOutput :
        fileOutput.write(name + ' best params: ' + str(best) + '\n')


def save_model_xgb(model , pm , model_name , name) :
    model_path = pm[ 'dir' ] + '/' + pm[ 'Project_name' ] + pm[
        'model_dir' ] + '/' + model_name + '_' + name + '_model.json'
    if not os.path.isfile(model_path) :
        model.save_model(model_path)


def plot_hyperopt_results(best , trials , pm , model_name , name) :
    parameters = best.keys()
    num_params = len(parameters)
    # make figsize dependent on number of parameters
    figsize = (num_params * 5 , 5)

    f , axes = plt.subplots(ncols=num_params , figsize=figsize)
    cmap = plt.cm.jet
    num_trials = len(trials.trials)

    for i , param in enumerate(parameters) :
        xs = np.array([ t[ 'misc' ][ 'vals' ][ param ] for t in trials.trials ]).ravel()
        ys = [ -t[ 'result' ][ 'loss' ] for t in trials.trials ]

        for j in range(num_trials) :
            color = cmap(float(j) / num_trials)
            axes[ int(i) ].scatter(xs[ j ] , ys[ j ] , s=20 , linewidth=0.01 , alpha=0.5 , color=color)

        axes[ int(i) ].set_title(param)

    plt.savefig(
        pm[ 'dir' ] + '/' + pm[ 'Project_name' ] + pm[ 'fig_dir' ] + '/' + model_name + '_%s_hyp_search.png' % name)


def ecpf4_featurizer(data) :
    """
    Compute the ECPF4 fingerprints for the input data.

    Parameters:
    - data: DataFrame, the data containing SMILES strings.

    test_mol = Chem.MolFromSmiles(test_compound)
    test_bit = AllChem.GetMorganFingerprintAsBitVect(test_mol, radius=2, nBits=2048)"""

    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import DataStructs

    ecpf4 = [ ]
    for index , row in data.iterrows() :
        smiles = row[ 'SMILES' ]
        mol = Chem.MolFromSmiles(smiles)
        bit = AllChem.GetMorganFingerprintAsBitVect(mol , radius=2 , nBits=2048)
        # Convert the ExplicitBitVect to a numpy array
        arr = np.zeros((1 ,) , int)
        DataStructs.ConvertToNumpyArray(bit , arr)
        ecpf4.append(arr)
    # return as dataframe
    return pd.DataFrame(ecpf4)


from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
import numpy as np


def evaluate_regression_model(y_true , y_pred , model=None , X_test=None , y_test=None) :
    results = { }

    # In-sample evaluation metrics
    results[ 'MAE' ] = mean_absolute_error(y_true , y_pred)
    results[ 'MSE' ] = mean_squared_error(y_true , y_pred)
    results[ 'RMSE' ] = np.sqrt(results[ 'MSE' ])
    results[ 'R-squared' ] = r2_score(y_true , y_pred)

    # Adjusted R-squared (for multiple regression, if a model is provided)
    if model is not None and X_test is not None :
        n = X_test.shape[ 0 ]  # Number of observations
        p = X_test.shape[ 1 ]  # Number of features
        adj_r_squared = 1 - (1 - results[ 'R-squared' ]) * (n - 1) / (n - p - 1)
        results[ 'Adjusted R-squared' ] = adj_r_squared

    # Out-of-sample evaluation (if a model and test dataset are provided)
    if model is not None and X_test is not None and y_test is not None :
        y_test_pred = model.predict(X_test)
        results[ 'Test MAE' ] = mean_absolute_error(y_test , y_test_pred)
        results[ 'Test MSE' ] = mean_squared_error(y_test , y_test_pred)
        results[ 'Test RMSE' ] = np.sqrt(results[ 'Test MSE' ])
        results[ 'Test R-squared' ] = r2_score(y_test , y_test_pred)

    # MAPE (if applicable, avoiding division by zero)
    if np.any(y_true) :
        results[ 'MAPE' ] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return results


def testtheresults(y_true , y_predicted , scores_pred_test) :
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import roc_auc_score

    # if y_predicted has more then 100 values which are equal to exactly 1 then give warning
    if (y_predicted == 1).sum() > 100 :
        print("Warning: y_predicted has more then 100 values which are equal to exactly 1")
    # if y_predicted has more then 100 values which are equal to exactly 0 then give warning
    if (y_predicted == 0).sum() > 100 :
        print("Warning: y_predicted has more then 100 values which are equal to exactly 0")

    auc = roc_auc_score(y_true , y_predicted[ : , 1 ])
    tn , fp , fn , tp = confusion_matrix(y_true , scores_pred_test , labels=[ 0 , 1 ]).ravel()
    sensitivity = tp / (tp + fn)
    specicifity = tn / (tn + fp)
    MCC = matthews_corrcoef(y_true , scores_pred_test)
    r = { 'sensitivity' : sensitivity , 'specicifity' : specicifity , 'AUC' : auc , 'MCC' : MCC }
    return r


def Do_XGradientBoost(X , y , testSet , scores_test , name , pm , scored , seed) :
    # Initialize MinMaxScaler
    m = 'XGB'
    scaler = MinMaxScaler()

    # Fit and transform X and transform testSet
    X = scaler.fit_transform(X)
    testSet_norm = scaler.transform(testSet)
    # Save normalizing parameters
    scaler_path = pm[ 'dir' ] + '/' + pm[ 'Project_name' ] + pm[ 'model_dir' ] + '/' + name + '_scaler.pkl'
    joblib.dump(scaler , scaler_path)

    # save normalizing parameters

    if pm[ 'hyp_tune' ] == 1 :
        print(
            "|---------------------------------------------------\nStarting Hyp Search for XGBoost Classifier\n|---------------------------------------------------")

        X_tune , _ , y_tune , _ = train_test_split(X , y , test_size=0.2 , random_state=seed , shuffle=False)

        def hyperopt_train_test(params) :
            model = XGBClassifier(**params , random_state=seed)
            return cross_val_score(model , X_tune , y_tune , cv=5 , n_jobs=8).mean()

        space4xgb = { 'n_estimators' : scope.int(hp.quniform('n_estimators' , 5 , 300 , 1)) ,
                      'learning_rate' : hp.choice('learning_rate' , [ 0.03 , 0.06 , 0.12 , 0.25 , 0.5 ]) ,
                      'max_depth' : scope.int(hp.quniform('max_depth' , 5 , 20 , 1)) ,
                      'min_child_weight' : scope.int(hp.quniform('min_child_weight' , 1 , 20 , 1)) ,
                      'gamma' : scope.int(hp.quniform('gamma' , 0 , 9 , 1)) }

        best_score = 0
        best_params = None

        def f(params) :
            nonlocal best_score
            nonlocal best_params
            acc = hyperopt_train_test(params)
            if acc > best_score :
                best_score = acc
                best_params = params
                print('new best:' , best_score , params)
            return { 'loss' : -acc , 'status' : STATUS_OK }

        trials = Trials()
        best = fmin(f , space4xgb , algo=tpe.suggest , max_evals=pm[ 'maxevals' ] , trials=trials)

        # Print the best parameters
        print("Best parameters:" , best_params)

        log_best_params(name , best_params , pm)

        plot_hyperopt_results(best_params , trials , pm , 'XGB' , name)

    else :
        best_params = { 'gamma' : 0 , 'learning_rate' : 0.25 , 'max_depth' : 20 , 'min_child_weight' : 4 ,
                        'n_estimators' : 200 }

    model = XGBClassifier(**best_params , random_state=seed)
    scored[ 'cv' ] = str(cross_val_score(model , X , y , scoring='accuracy' , cv=5 , n_jobs=8))
    model.fit(X , y)

    print('5-fold cross validation: ' , scored[ 'cv' ])
    scores_prob_test = model.predict_proba(testSet_norm)
    scores_pred_test = model.predict(testSet_norm)

    scored[ 'score' ] = model.score(X=testSet_norm , y=scores_test)
    accuracy = accuracy_score(scores_test , scores_pred_test)
    print('Test Score: ' , scored , accuracy)

    save_model_xgb(model , pm , m , name)

    testresults = testtheresults(scores_test , scores_prob_test , scores_pred_test)
    score_xgb = { **scored , **testresults }
    score_xgb = pd.DataFrame.from_dict(score_xgb , orient='index' , columns=[ name ])
    save_model_xgb(model , pm , 'XGBc' , name)
    return score_xgb
