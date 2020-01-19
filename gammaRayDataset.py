import time
import numpy as np

import random 

from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn import svm

np.random.seed(80)#80577904

#print('\nGamma Ray dataset')
x = []
y = []
infile = open("magic04.txt","r")
for line in infile:
    y.append(int(line[-2:-1] =='g'))
    x.append(np.fromstring(line[:-2], dtype=float,sep=','))
infile.close()
x = np.array(x).astype(np.float32)
y = np.array(y) 
#Split data into training and testing
ind = np.random.permutation(len(y))
split_ind = int(len(y)*0.8)
x_train = x[ind[:split_ind]]
x_test = x[ind[split_ind:]]
y_train = y[ind[:split_ind]]
y_test = y[ind[split_ind:]]
############################### KNN ###########################################
#print('\nKNN')
model_KNN = KNeighborsClassifier(weights='distance')
start = time.time()
model_KNN.fit(x_train, y_train)
predTrain = model_KNN.predict(x_train)       
pred = model_KNN.predict(x_test)   
elapsed_time = time.time()-start
print('{0:.6f} '.format(elapsed_time))  
print((np.sum(predTrain==y_train)/len(y_train))*100)   
print((np.sum(pred==y_test)/len(y_test))*100)
############################### Decision Tree #################################
print('\nDecision Tree')
model_DT = tree.DecisionTreeClassifier(criterion='entropy')
#criterion='entropy',max_depth = 34,min_samples_split=2, splitter= 'best'
start = time.time()
model_DT.fit(x_train, y_train) 
predTrain = model_DT.predict(x_train)     
pred = model_DT.predict(x_test)   
elapsed_time = time.time()-start
print('{0:.6f} '.format(elapsed_time)) 
print((np.sum(predTrain==y_train)/len(y_train))*100)   
print((np.sum(pred==y_test)/len(y_test))*100,"\n")
############################### Random Forests ################################
#print('\nRandom Forests')
model_RF = RandomForestClassifier(n_estimators=10,max_features='log2')
#n_estimators=65 , criterion='entropy', max_features='auto' , max_depth = none,min_samples_split=2
start = time.time()
model_RF.fit(x_train, y_train)
predTrain = model_RF.predict(x_train)   
pred = model_RF.predict(x_test)   
elapsed_time = time.time()-start
print('{0:.6f} '.format(elapsed_time))   
print((np.sum(predTrain==y_train)/len(y_train))*100)     
print(np.sum(pred==y_test)/len(y_test)*100)
############################### Logistic Regression ###########################
#print('\nLogistic Regression')
model_LR = LogisticRegression()
#fit_intercept,solver='lbfgs',multi_class='multinomial' 
start = time.time()
model_LR.fit(x_train, y_train)  
predTrain = model_LR.predict(x_train)   
pred = model_LR.predict(x_test)   
elapsed_time = time.time()-start
print('{0:.6f} '.format(elapsed_time))   
print((np.sum(predTrain==y_train)/len(y_train))*100)     
print(np.sum(pred==y_test)/len(y_test)*100)
############################### SVM ###########################################
#print('\nSVM')
model_svm = svm.SVC(gamma='scale',C=8,kernel='rbf')
#gamma='scale',C=3.9, degree=3,kernel
start = time.time()
model_svm.fit(x_train, y_train)    
predTrain = model_svm.predict(x_train)   
pred = model_svm.predict(x_test)   
elapsed_time = time.time()-start
print('{0:.6f} '.format(elapsed_time))   
print((np.sum(predTrain==y_train)/len(y_train))*100)     
print(np.sum(pred==y_test)/len(y_test)*100)
