import os
from collections import Counter

from imblearn.over_sampling import SMOTE

from Do_ML import Do_XGradientBoost_regression, ecpf4_featurizer, Do_XGradientBoost, maccs_featurizer

pm = { 'Project_name' : 'March11_CACHE5' ,
       'dir' : 'CACHE5/' ,
       'data_dir' : 'Clusters_Max_TC' ,
       'data_file' : '/train_set_March4_CACHE5_',
       "model_dir" : '/models' ,
       'fig_dir' : '/figs' ,
       'num_test_set_clusters' : 50 ,  # number of clusters to make test set from
       'test_set_cluster_size' : 10 ,  # number of molecules to take from each cluster for test set
       'use_some_data' : 1 ,  # Use only 100 molecules for testing
       'maxevals' : 30 ,  # Number of evaluations for hyperopt
       'output_filename' : '/home/weiser/PYTHON/CACHE5/results.txt' ,  # File to write the results to
       'tan' : [ 0.3, 0.4 , 0.6 , 0.8 ] ,
       'hyp_tune' : 1,
       'regr' : 1,
       'class' : 0,
       }
#TODO make a better system for directory pointing

# make model and fig directories
model_dir =pm[ 'Project_name' ] + pm[ 'model_dir' ]
fig_dir =pm[ 'Project_name' ] + pm[ 'fig_dir' ]
os.makedirs(model_dir , exist_ok=True)
os.makedirs(fig_dir , exist_ok=True)
seed = 42

#load test set
import pandas as pd
test_set = pd.read_csv(pm['data_dir'] + '/test_setMarch4_CACHE5.csv')
print('test_set.shape:', test_set.shape)
x_test = test_set.drop('pAct' , axis=1)
x_test = maccs_featurizer(x_test)
y_test = test_set['pAct']
y_test_class = pd.Series(0, index=y_test.index)  # Initialize with zeros
y_test_class[y_test > 8] = 1

a = {}
result_all = pd.DataFrame()


for tc in pm['tan']:
       name = str(tc)
       print('training model for tan = ', tc)
       train_data = pd.read_csv(pm['data_dir'] + pm[ 'data_file' ] + str(tc) + '.csv')
       if pm[ 'use_some_data' ] == 1:
           train_data = train_data.sample(n=200)
       print('train_data.shape:', train_data.shape)

       x_train = train_data.drop('pAct' , axis=1)
       x_train = maccs_featurizer(x_train)
       y_train = train_data['pAct']

       if pm['regr'] == 1:
              result = Do_XGradientBoost_regression(x_train, y_train, x_test, y_test, name, pm, a, seed)
       if pm['class'] == 1:
              #create binary labels ; 1 if pAct > 8, 0 if pAct < 7.5, discard if in the midle
              train_data[ 'binary_label' ] = 0  # Initialize column
              train_data.loc[ train_data[ 'pAct' ] > 8 , 'binary_label' ] = 1

              #to discard middle
              #train_data = train_data[ (train_data[ 'pAct' ] > 8) | (train_data[ 'pAct' ] < 7.5) ]  # Discard middle

              x_train_class = ecpf4_featurizer(train_data.drop([ 'pAct' , 'binary_label' ] , axis=1))
              y_train_class = train_data[ 'binary_label' ]

              counter = Counter(y_train_class)

              if counter[ 0 ] > counter[ 1 ] :
                     ada = SMOTE(random_state=seed)
                     x_train_class , y_train_class = ada.fit_resample(x_train_class , y_train_class)
              counter_aft = Counter(y_train_class)
              print('Before SMOTE' , counter)
              print('After SMOTE' , counter_aft)

              result = Do_XGradientBoost(x_train_class, y_train_class, x_test, y_test_class, name, pm, a, seed)

       print(result)
       result_all = pd.concat([result_all, result], axis=1)

result_all.to_csv(pm['Project_name'] + '/results_all.csv')
print(result_all)
#


