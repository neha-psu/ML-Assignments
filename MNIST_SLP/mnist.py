import os
import pylab as py
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import copy
learningrate=0.1
#learningrate=0.01
#learningrate=0.001
train_data_ori= pd.read_csv("mnist_train.csv")
train_temp=train_data_ori.values
accuracy_result_train=[]
train_data = train_temp[:, 1:]
train_data = train_data / 255
bias = 1
train_data = np.insert(train_data,0,bias, axis=1)
train_label = train_temp[:, :1]
weights_train = np.random.uniform(-0.05, 0.05, size=(785, 10))
prediction_cfm=np.array([])
noofclassses=10
cfm_train_target = np.array(train_label[:,0].astype(int)).reshape(-1)
one_hot_targets = np.eye(noofclassses)[cfm_train_target]

for epoch in range(0,71):
    correctprediction = 0
    for i in range(0,59999):
        weightupdate = False
        prediction = np.dot(train_data[i], weights_train)
        # np.argmax gives max of all values along the row
        max = np.argmax(prediction)
        #one hot encoding
        target = np.zeros((1, 10))
        np.put(target, train_label[i].astype(int), 1)
        if epoch==70:
            cfm_test_predicted = np.zeros((1, 10))
            np.put(cfm_test_predicted, max, 1)
            if prediction_cfm.size==0:
                prediction_cfm=np.hstack((prediction_cfm,np.array(cfm_test_predicted[0])))  # np.hstack horizontally stacks the 2 arrays
            else:
                prediction_cfm=np.vstack((prediction_cfm,np.array(cfm_test_predicted[0])))
        if (max != train_label[i]):
            weightupdate = True
        else:
            correctprediction = correctprediction + 1
        if (weightupdate==True and epoch!=0):
            prediction[prediction > 0] = 1
            prediction[prediction <= 0] = 0            
            ty = target - prediction
            for col in range(0, 10):
                    temp=ty[:, col] * train_data[i]
                    testing = np.matrix(temp)
                    testing=np.transpose(testing)
                    testing2=np.matrix(learningrate * testing[:,0])
                    testing2_temp=np.array(testing2[:,0])
                    add=np.asmatrix(weights_train[:, col] + testing2_temp[:,0])
                    weights_train[:,col]=add                    
    accuracy = (correctprediction / 60000)*100
    accuracy_result_train=np.append(accuracy_result_train,accuracy)
    if epoch==0:
        print("Epoch 0 Accuracy of Train Data: ",accuracy_result_train.astype(int))


test_data_ori= pd.read_csv("mnist_test.csv")
test_temp=test_data_ori.values
accuracy_result_test=[]
test_data = test_temp[:, 1:]
test_data = test_data / 255
bias = 1
test_data = np.insert(test_data,0,bias, axis=1)
test_label = test_temp[:, :1]
weights_test = np.random.uniform(-0.05, 0.05, size=(785, 10))
prediction_cfm=np.array([])
noofclassses=10
cfm_test_target = np.array(test_label[:,0].astype(int)).reshape(-1)
one_hot_targets = np.eye(noofclassses)[cfm_test_target]

for epoch in range(0,71):
    correctprediction = 0
    for i in range(0,9999):
        weightupdate = False
        prediction = np.dot(test_data[i], weights_test)
        max = np.argmax(prediction)
        target = np.zeros((1, 10))
        np.put(target, test_label[i].astype(int), 1)
        if epoch==70:
            cfm_test_predicted = np.zeros((1, 10))
            np.put(cfm_test_predicted, max, 1)
            if prediction_cfm.size==0:
                prediction_cfm=np.hstack((prediction_cfm,np.array(cfm_test_predicted[0])))
            else:
                prediction_cfm=np.vstack((prediction_cfm,np.array(cfm_test_predicted[0])))
        if (max != test_label[i]):
            weightupdate = True
        else:
            correctprediction = correctprediction + 1
        if (weightupdate==True and epoch!=0):
            prediction[prediction > 0] = 1
            prediction[prediction <= 0] = 0            
            ty = target - prediction
            for col in range(0, 10):
                    temp=ty[:, col] * test_data[i]
                    testing = np.matrix(temp)
                    testing=np.transpose(testing)
                    testing2=np.matrix(learningrate * testing[:,0])
                    testing2_temp=np.array(testing2[:,0])
                    add=np.asmatrix(weights_test[:, col] + testing2_temp[:,0])
                    weights_test[:,col]=add                    
    accuracy = (correctprediction / 10000)*100
    accuracy_result_test=np.append(accuracy_result_test,accuracy)
    if epoch==0:
        print("Epoch 0 Accuracy of test Data: ",accuracy_result_test.astype(int)) 
               
print("Accuracy of Train Data: ",accuracy_result_train)
cfm=confusion_matrix(one_hot_targets.argmax(axis=1),prediction_cfm.argmax(axis=1))
print("Confusion Matrix for Train data: ")
print(cfm)
plt.plot(accuracy_result_train,color='green',label='Train Data Accuracy')
plt.ylabel("Accuracy in %")
plt.xlabel("Epoch")

print("Accuracy of Test Data: ")
print(accuracy_result_test)
cfm=confusion_matrix(one_hot_targets.argmax(axis=1),prediction_cfm.argmax(axis=1))
print("Confusion Matrix for Test data ")
print(cfm)
plt.plot(accuracy_result_test,color='red',label='Test Data Accuracy')


image= "learningrate_1.png"
plt.title("learning rate 0.1")
plt.savefig(image)
plt.show()
