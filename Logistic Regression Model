import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import sklearn as sci
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
from keras.models import model_from_json
from keras.layers import Dropout
from keras import regularizers
from collections import Counter
from heapq import merge
import copy
import random

data=pd.read_csv('C:###/###/neural_data.csv')

data.head()

##predictor value is the binary predictors for the training/test data
sns.countplot(data=data,x='predictor value')

data = data.dropna()

target = data.pop('predictor value')

##balancing the data here to make sure that there are equal numbers for both classes
zero_pos = np.where(np.array(target) == 0)[0]
one_pos = np.where(np.array(target) == 1)[0]
spots_I_need=list(merge(one_pos,zero_pos[0:len(one_pos)]))
data= data.iloc[spots_I_need,:]
target=target.iloc[spots_I_need]

##checking the predictor distribution
sns.countplot(data=target,x='predictor value')

##back to numpy form
target=target.to_numpy()
data=data.to_numpy()
random.seed(10)

##if not done this will throw an error with a (n,1)
target=target.ravel()

##splitting the data into training and test sets  of a general size and randomness of choice
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5,random_state=7)

##general model to train on
model = Sequential()
model.add(Dense(20, activation='sigmoid', kernel_initializer='he_normal', input_shape=(8,)))
model.add(Dense(500, activation='sigmoid', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=0)
loss, acc = model.evaluate(X_test, y_test, verbose=0)

##visualizing model performance
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='train_loss')
pyplot.plot(history.history['accuracy'], label='train_accuracy')
pyplot.legend()
pyplot.show()

##accuracy of the neural model
print('Test Accuracy: %.3f' % acc)

#########################################################################LOGISTIC REGRESSION MODEL##########################################################

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

##setting the max iterations to above 100 which is standard so as to not incur an error
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)

# fit the model with data
logreg.fit(X_train,y_train)

#testing against the tested data to form a prediction
y_pred=logreg.predict(X_test)

##confusion matrix to check the accuracy
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


##necessary to visualize a heatmap for the confusion matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

##the accuracy of the model
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

##the precision of the model
print("Precision:",metrics.precision_score(y_test, y_pred)

##the recall of the model
print("Recall:",metrics.recall_score(y_test, y_pred))

##roc/auc section
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

##specifying to see what the performance would be with all zeros
ns_probs = [0 for _ in range(len(y_test))]

#fitting the model 
model = LogisticRegression(solver='lbfgs', max_iter=1000)

##training the model
model.fit(X_train,y_train)

#predicted probabilities
lr_probs = model.predict_proba(X_train)

##probabilities of the positive results
lr_probs = lr_probs[:, 1]

#roc/auc scores for the all zero model as well as the actual tested model
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_train, lr_probs)

##visualizing the scores for each
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

##curves for plotting
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_train, lr_probs)

# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


