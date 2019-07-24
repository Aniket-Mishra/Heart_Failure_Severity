# Importing the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential # used to initialize NN
from keras.layers import Dense # model to create different layers in NN

def dataset_path() :
    abs_path = os.path.dirname(os.path.abspath(__file__)) 
    path = os.path.join(abs_path, 'Datasets')
    path = os.path.join(path, 'Heart.csv')
    return path

# Importing the dataset
dataset = pd.read_csv(dataset_path())
X = dataset.iloc[:, 0:13].values #index of columns in the independent (predictor) variables
y = dataset.iloc[:, 13:18].values #col 13 (what we are predicting)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling - MUST scale for any NN model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Making the ANN! 

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
# dense helps to put an initial weight (needs to start somewhere)
# add (layer) will add a layer
# 6 nodes in the hidden layer (tip: input nodes + output nodes /2), and tells next layer no. of nodes to expect
# uniform is to randomly initialize the weights to a uniform distribution
# activation is the function you will use (relu is rectifier)
# input dim --> number of inputs from input layer

# Adding the second hidden layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
# knows what inputs to expect because there is already an input layer created

# Adding the output layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# using adam optimizer --> algorithm to use to find the optimal weights
# loss: need to have a loss function (which you are trying to minimize), binary crossentropy for binary output
                                     
# Fitting the ANN to the Training set and training the ANN
classifier.fit(X_train, y_train, batch_size = 15, epochs = 100)

y_pred = classifier.predict(X_test) # gives prediction for each observation in test set


'''checking HF severity for a new row: e.g. patient with:
Age: 54
Sex: Male
Cp: 4
Trestbps: 168
Cholesterol: 350
Fbs: no (0)
RestECG: 2
Thalach: 167
Exang: yes (1)
Oldpeak: 2.8
Slope: 2
Ca: 2
Thal: 7
'''

sample_patient = sc.transform(np.array([[54,1,4,168,350,0,2,167,1,2.8,2,2,7]]))
sample_pred = classifier.predict(sample_patient)
'''
For sample patient listed above: % chance of heart failure at different severity
    level 0 = 12.4%
    level 1 = 24%
    level 2 = 32.2%
    level 3 = 35.9%
    level 4 = 12.5%
'''