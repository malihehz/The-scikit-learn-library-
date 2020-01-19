import time
import numpy as np
import mnist

from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn import svm

np.random.seed(80)

#print('MNIST dataset')
skip = 50
x_train, y_train, x_test, y_test = mnist.load()
x_train = np.array(x_train[::skip], dtype=np.int)
y_train = np.array(y_train[::skip], dtype=np.int)
x_test = np.array(x_test[::skip], dtype=np.int)
y_test = np.array(y_test[::skip], dtype=np.int)
############################### KNN ###########################################
#print('\nKNN')
model_KNN = KNeighborsClassifier(weights='distance',p=2)    
#weights='distance',p=2
start = time.time()
model_KNN.fit(x_train, y_train)
elapsed_time = time.time()-start
#print('{0:.6f} '.format(elapsed_time)) 
#start = time.time()  
predTrain = model_KNN.predict(x_train)     
pred = model_KNN.predict(x_test)   
elapsed_time = time.time()-start
print('{0:.6f} '.format(elapsed_time))  
print((np.sum(predTrain==y_train)/len(y_train))*100)   
print((np.sum(pred==y_test)/len(y_test))*100,"\n")
############################### Decision Tree #################################
#print('\nDecision Tree')
model_DT = tree.DecisionTreeClassifier(criterion='entropy')
#criterion='entropy',max_depth = 13,min_samples_split=2, splitter= 'best'
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
model_RF = RandomForestClassifier(n_estimators=100,criterion='entropy')
#n_estimators=30 , criterion='entropy', max_features='auto' , max_depth = none,min_samples_split=2
start = time.time()
model_RF.fit(x_train, y_train) 
predTrain = model_RF.predict(x_train)    
pred = model_RF.predict(x_test)   
elapsed_time = time.time()-start
print('{0:.6f} '.format(elapsed_time)) 
print((np.sum(predTrain==y_train)/len(y_train))*100)      
print((np.sum(pred==y_test)/len(y_test))*100,"\n")
############################### Logistic Regression ###########################
#print('\nLogistic Regression')
model_LR = LogisticRegression(solver='sag',multi_class='multinomial')
#fit_intercept,solver='newton-cg',multi_class='multinomial'
start = time.time()
model_LR.fit(x_train, y_train)  
predTrain = model_LR.predict(x_train)    
pred = model_LR.predict(x_test)   
elapsed_time = time.time()-start
print('{0:.6f} '.format(elapsed_time)) 
print((np.sum(predTrain==y_train)/len(y_train))*100)      
print((np.sum(pred==y_test)/len(y_test))*100,"\n")
############################### SVM ###########################################
#print('\nSVM')
model_svm = svm.SVC(gamma='scale',C=3.9,kernel='rbf')
#gamma='scale',C=3.9, degree=3
start = time.time()
model_svm.fit(x_train, y_train) 
predTrain = model_svm.predict(x_train)    
pred = model_svm.predict(x_test)   
elapsed_time = time.time()-start
print('{0:.6f} '.format(elapsed_time)) 
print((np.sum(predTrain==y_train)/len(y_train))*100)      
print((np.sum(pred==y_test)/len(y_test))*100,"\n")
