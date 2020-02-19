import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

### Preprocessing ---- 1. Create training and test set:
data = np.genfromtxt('spambase.data', delimiter=',')
np.random.shuffle(data)
## grab the last column as target and rest as features
target = data[: ,-1]
features = data[:,0:-1]
total_features=features.shape[1]
## standerize the features --- mean = 0  and S.D =1
features = preprocessing.scale(features);
## Splitting training and test dataset
train_data, test_data, train_target, test_target = train_test_split(features, target, test_size=0.5, random_state=42)

print(train_data.shape, test_data.shape, train_target.shape, test_target.shape)

###prior probability for each class, 
###1 (spam) and 0 (not-spam) in the training data.
#### P(spam) should be approx 0.4 and P(not-spam) should be 0.6
train_data_length = len(train_data)
total_spam = 0
for i in range (train_data_length):
    if train_target[i] == 1 :
        total_spam +=1
        
probability_spam = total_spam/train_data_length
probability_not_spam = 1 - probability_spam
print("The probability of spam mail and not spam mail respectively is: ", probability_spam , probability_not_spam)


### Mean and Standard deviation for each feature
mean_of_Spam =[]
SD_spam =[]
mean_NotSpam =[]
SD_NotSpam =[]
for i in range (0,total_features):
    spam=[]
    NotSpam=[]
    for j in range (0,train_data_length):
        if train_target[j] == 1 :
            spam.append(train_data[j][i])
        else :
            NotSpam.append(train_data[j][i])
    
    
    mean_of_Spam.append(np.mean(spam))
    SD_spam.append(np.std(spam))
    mean_NotSpam.append(np.mean(NotSpam))
    SD_NotSpam.append(np.std(NotSpam))

###avoid a divide-by-zero error in Gaussian Naïve Bayes. (replace 0 SD with 0.0001 )
for i in range (0, total_features) :
    if(SD_spam[i] == 0):
        SD_spam[i]  =0.0001
    if(SD_NotSpam[i]==0):
        SD_NotSpam[i] = 0.0001
        
#### Gaussian naive bayes implementation  on test data
predicted =[]
for i in range (0, len(test_data)):
    spam_test =[] ##probabilty(test_data[j]/class=spam)
    NotSpam_test =[]##probabilty(test_data/class=not spam)
    for j in range (0, total_features):
        x1=float(1)/((np.sqrt(2*np.pi))*SD_spam[j])
        x2=np.power(np.e, -(np.square(test_data[i][j] - mean_of_Spam[j])/(2* np.square(SD_spam[j]))))
        spam_test.append(x1*x2)
        x3=float(1)/((np.sqrt(2*np.pi))*SD_NotSpam[j])
        x4=np.power(np.e, -(np.square(test_data[i][j] - mean_NotSpam[j])/(2* np.square(SD_NotSpam[j]))))
        NotSpam_test.append(x3*x4)

    spam_probability_testdata = np.log(probability_spam) + np.sum(np.log(np.asarray(spam_test)))
    Notspam_probability_testdata = np.log(probability_not_spam) + np.sum(np.log(np.asarray(NotSpam_test)))
    predicted.append(np.argmax([Notspam_probability_testdata, spam_probability_testdata]))        

###calculate confusion_matrix, precision, Recall and accuracy.
accuracy = accuracy_score(test_target, predicted)
precision = precision_score(test_target, predicted)
recall = recall_score(test_target, predicted)

test_confusion_matrix = confusion_matrix(test_target,predicted)

print("Confusion matrix:\n",test_confusion_matrix)
print ("Accuracy Value: ",accuracy)
print ("Precision Value: ", precision)
print ("Recall Value: ",recall)      
