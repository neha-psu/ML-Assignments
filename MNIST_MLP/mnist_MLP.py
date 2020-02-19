import csv
import numpy as np
import pandas as pd
import math
from scipy.special import expit
from sklearn.metrics import confusion_matrix
import random

class perceptron_mnist_MLP:
    
    def __init__(self, n, train_csv, test_csv):
        self.bias=1
        self.n=n
        self.train_data = pd.read_csv(train_csv)
        self.train_data = self.train_data.to_numpy()
        print("train_data shape:", self.train_data.shape)
        self.test_data = pd.read_csv(test_csv)
        self.test_data = self.test_data.to_numpy()
        print("test_data shape:", self.test_data.shape)
        
        # For exp3: randomly shuffle the train_data in 1:4
        #np.random.shuffle(self.train_data)
        #self.train_data = self.train_data[0:15000]a9
        #print(self.train_data.shape)
        self.re_init(self.n)
        
    def re_init(self,n):
        self.weight_ItoH = np.random.uniform(-0.05,0.05,(785,n)) # Here n is number of hidden inputs
        self.weight_HtoO = np.random.uniform(-0.05,0.05,((n+1),10)) # Here n+1 is number of hidden inputs + bias
        
        #store previous weights from input to hidden
        self.prev_weight_ItoH = np.zeros((785, n))
        
        #store previous weights from hidden to output
        self.prev_weight_HtoO = np.zeros(((n+1),10))
        
        # initialize the train and test accuracy as empty array
        self.train_accuracy=[]
        self.test_accuracy = [] 
        
        #store the  hidden activation
        self.hidden_input = np.zeros((1,n+1))
        self.hidden_input[0,0] = 1 # include one bias input =1
        
    def learn(self, epoch, eta, momentum):
        self.eta=eta
        self.momentum=momentum
        for each in range(epoch):
            self.perceptron_MLP(each, self.train_data, 1)
            self.perceptron_MLP(each, self.test_data, 0)
        print("\nTrain data accuracy\n")
        print(self.train_accuracy)
        print("\nTest data accuracy\n")
        print(self.test_accuracy)
        
        
        
    def perceptron_MLP(self, epoch, input_data, flag):
        predicted = []
        actual = []
        for i in range(input_data.shape[0]):
            target_class = input_data[i,0].astype('int')
            actual.append(target_class)  
            xi = input_data[i]/255
            xi[0] = self.bias
            xi = xi.reshape(1,785)
            
            sigmoid_hidden = expit(np.dot(xi, self.weight_ItoH))
            self.hidden_input[0,1:] = sigmoid_hidden
            
            sigmoid_output = expit(np.dot(self.hidden_input, self.weight_HtoO))
            predict_output = np.argmax(sigmoid_output)
            predicted.append(predict_output)
            
            ##calculating error and weight updation for training dataset
            if epoch>0 and flag==1:
                # 0.9 hot enconding of target_class
                target_output = np.zeros((1,10)) + 0.1
                target_output[0,target_class]=0.9
                
                # error calculation for hidden and output layer
                error_output = sigmoid_output * (1-sigmoid_output) * (target_output - sigmoid_output)
                error_hidden = sigmoid_hidden * (1-sigmoid_hidden) * (np.dot(error_output, self.weight_HtoO[1:,:].T))
                
                # weight updation for output layer
                delta_weight_HtoO = (self.eta * np.dot(self.hidden_input.T, error_output)) + (self.momentum * self.prev_weight_HtoO)
                self.prev_weight_HtoO = delta_weight_HtoO
                self.weight_HtoO += delta_weight_HtoO
                
                # weight updation for hidden layer
                delta_weight_ItoH = (self.eta * np.dot(xi.T, error_hidden)) + (self.momentum * self.prev_weight_ItoH)
                self.prev_weight_ItoH = delta_weight_ItoH
                self.weight_ItoH += delta_weight_ItoH
        
        if flag ==1:
            train_confusion_matrix = confusion_matrix(actual,predicted);
            train_diagonalsum = sum(np.diag(train_confusion_matrix))
            train_accur = (train_diagonalsum/float(input_data.shape[0])) * 100
            #train_accur = (np.array(predicted) == np.array(actual)).sum()/len(actual)*100
            print("for epoch ",epoch)
            print("Train accuracy: ",train_accur)
            self.train_accuracy.append(train_accur)
            if(epoch==49):
                print("\nConfusion matrix for train data \n")
                print(train_confusion_matrix)
            
        if flag== 0:
            test_confusion_matrix = confusion_matrix(actual,predicted);
            test_diagonalsum = sum(np.diag(test_confusion_matrix))
            test_accur = (test_diagonalsum/float(input_data.shape[0])) * 100
            #test_accur = (np.array(predicted) == np.array(actual)).sum()/len(actual)*100
            print("Test accuracy: ",test_accur)
            self.test_accuracy.append(test_accur)
            if(epoch==49):
                print("\nConfusion matrix for test data \n")
                print(test_confusion_matrix)
            
        

    
            
                

        