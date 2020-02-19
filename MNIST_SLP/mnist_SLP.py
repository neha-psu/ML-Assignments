import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as   plt
import csv
import random
#from sklearn.metrics import confusion_matrix



class perceptron_mnist:
    
    def cost_function(self):
        return np.sum(np.square(self.target_value - self.outputs))
    
    def sanitize_output(self, output):
        ret_output = output.copy()
        ret_output[ret_output > 0] = 1
        ret_output[ret_output <= 0] = 0
        return ret_output
    
    def confuse_matrix(self, actual, predicted):
     c_matrix = np.zeros((actual.shape[1], actual.shape[1]))
     # Reverse of one hot encoding for actual and predicted value i.e. find the index of maximum value along the axis)
     # (Here axis =1 means along the row. and axis=0  means along the column)
     actual = np.argmax(actual, axis=1) 
     predicted = np.argmax(predicted, axis=1) 
     #conf_matrix = confusion_matrix(actual, predicted)
     #print(conf_matrix.astype(int))
     for i in range(actual.shape[0]):
        c_matrix[actual[i]][predicted[i]] += 1
        
     return c_matrix
    
    def learn(self, epochs, eta):
        #self.outputs = np.zeros(self.target_value.shape)
        transposed_input = np.transpose(self.train_data)
                
        for epoch in range(epochs):
            self.outputs = np.zeros(self.target_value.shape)
            # if epoch <10:
                # #....pridict the output : y = a(w.x)
                # self.outputs = np.dot(self.train_data, self.weight_arr)
                # self.sanitize_output(self.outputs)
                # #....weight updation for each epoch (w : = w - eta(y-t)x)
                # self.weight_arr -= eta * (np.dot(transposed_input,(self.outputs - self.target_value)))
                # print("for epoch ",epoch)
                # print(self.accuracy(self.target_value, self.predict_output(self.outputs)))
                # self.train_accuracy.append(self.accuracy(self.target_value, self.predict_output(self.outputs)))
                
                # print(self.accuracy(self.test_target_value, self.predict()))
                # self.test_accuracy.append(self.accuracy(self.test_target_value, self.predict()))

            # else:   
            for input_data_idx in range(self.train_data.shape[0]):
                outputs = np.dot(self.train_data[input_data_idx], self.weight_arr)
                outputs = self.sanitize_output(outputs)
                input_shape = (self.train_data.shape[1],1)
                output_shape = (1,self.target_value.shape[1])
                self.weight_arr -= eta * (np.dot(self.train_data[input_data_idx].reshape(input_shape),(outputs - self.target_value[input_data_idx]).reshape(output_shape)))
                self.outputs[input_data_idx] = outputs
        
            print("for epoch ",epoch)
            print(self.accuracy(self.target_value, self.predict_output(self.outputs)))
            self.train_accuracy.append(self.accuracy(self.target_value, self.predict_output(self.outputs)))    
            print(self.accuracy(self.test_target_value, self.predict()))
            self.test_accuracy.append(self.accuracy(self.test_target_value, self.predict()))
    
    def accuracy(self, target, output):
        diff = np.square(target-output)
        diff_vector = np.sum(diff, axis =1)
        correct_values = np.count_nonzero(diff_vector == 0)
        total = target.shape[0]
        return 100*(correct_values/total)
    
    def predict_output(self, output):
        #np.argmax finds the index of maximum value along the axis
        indices = np.argmax(output, axis=1)
        ret_output = np.zeros(output.shape)
        for (i, index) in enumerate(indices):
            ret_output[i][index] = 1
        return ret_output.astype(int)
        
    def predict(self):
         #self.test_outputs = np.dot(self.test_inputs, self.weight_arr)
         #return self.predict_output(self.test_outputs)
         return self.predict_output(np.dot(self.test_inputs, self.weight_arr))
         #return self.sanitize_output(np.dot(self.test_inputs, self.weight_arr))
        
    def parse_input_csv_file(self, file):
        inputs0 = pd.read_csv(file)
        label=inputs0['label']
        label0=label.to_numpy()
        n_data=np.size(label0,0)
        # One hot encoding(10 is the number of output classes)
        target_value = np.zeros((n_data, 10))
        target_value[np.arange(n_data), label0] = 1
        inputs = inputs0.drop("label", axis=1)
        inputs = inputs.to_numpy()/255
        inputs = np.insert(inputs,0,self.bias, axis=1)
        
        return target_value.astype(int),inputs
    
    
    def re_init(self):
        self.weight_arr = np.random.uniform(-0.05,0.05,(785,10))
        # initialize the train and test accuracy as empty array
        self.train_accuracy=[]
        self.test_accuracy = [] 
        
        
    def __init__(self, train_csv, test_csv):
        self.bias = -1
        self.target_value, self.train_data = self.parse_input_csv_file(train_csv)
        self.test_target_value, self.test_inputs = self.parse_input_csv_file(test_csv)
        
        self.re_init()
  


