import time
import numpy as np

import random 

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
#from sklearn.datasets import make_regression

from sklearn.svm import SVR

np.random.seed(80)

#print('\nSolar particle dataset')
skip = 200
x_train = np.load('x_ray_data_train.npy')
y_train = np.load('x_ray_target_train.npy')
x_test = np.load('x_ray_data_test.npy')
y_test = np.load('x_ray_target_test.npy')
x_train = np.array(x_train[::skip])
y_train = np.array(y_train[::skip])
x_test = np.array(x_test[::skip])
y_test = np.array(y_test[::skip])

############################### KNN ###########################################
#print('\nKNN')
model_KNN = KNeighborsRegressor(weights='distance')#weights='distance',p=1
#weights='distance',p=1
#n_neighbors=9, weights='distance' algorithm='kd_tree' p=1 leaf_size = 30
start = time.time()
model_KNN.fit(x_train, y_train)    
predTrain = model_KNN.predict(x_train)  
pred = model_KNN.predict(x_test)
elapsed_time = time.time()-start
print('{0:.6f} '.format(elapsed_time))
print(np.mean(np.square(predTrain-y_train)))   
print(np.mean(np.square(pred-y_test)))
############################### Decision Tree #################################
#print('\nDecision Tree')
model_DT = DecisionTreeRegressor()
#criterion='mse',max_depth = 8,min_samples_split=2, splitter= 'best'
start = time.time()
model_DT.fit(x_train, y_train) 
predTrain = model_DT.predict(x_train)    
pred = model_DT.predict(x_test)
elapsed_time = time.time()-start
print('{0:.6f} '.format(elapsed_time))  
print(np.mean(np.square(predTrain-y_train)))   
print(np.mean(np.square(pred-y_test)))
############################### Random Forests ################################
#print('\nRandom Forests')
model_RF = RandomForestRegressor()
#n_estimators=30 , criterion='mse', max_features='auto' , max_depth = none,min_samples_split=2
start = time.time()
model_RF.fit(x_train, y_train)   
predTrain = model_RF.predict(x_train)
pred = model_RF.predict(x_test)
elapsed_time = time.time()-start
print('{0:.6f} '.format(elapsed_time))  
print(np.mean(np.square(predTrain-y_train))) 
print(np.mean(np.square(pred-y_test)))
############################### SVM ###########################################
#print('\nSVM')
model_svm = SVR(gamma='auto',C=4)
#gamma='auto',kernel='poly',degree=5
start = time.time()
model_svm.fit(x_train, y_train)  
predTrain = model_svm.predict(x_train)
pred = model_svm.predict(x_test)
elapsed_time = time.time()-start
print('{0:.6f} '.format(elapsed_time))  
print(np.mean(np.square(predTrain-y_train)))
print(np.mean(np.square(pred-y_test)))
