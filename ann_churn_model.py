# Part 1 - Preprocessing

# Classification Template

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[ :, 3:13].values
y = df.iloc[ :, -1].values

#Encoding categorical attributes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x1 = LabelEncoder()
X[:, 1] = le_x1.fit_transform(X[:, 1])
le_x2 = LabelEncoder()
X[:, 2] = le_x1.fit_transform(X[:, 2])
onehot = OneHotEncoder(categorical_features = [1])
X = onehot.fit_transform(X).toarray()
X = X[:, 1:] #to avoid dummy variable trap

#Splitting into training and testing set
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)


# Part 2 - Building the ANN

# Import keras libraries and packages
import keras
from keras.models import Sequential #to initialise the ANN
from keras.layers import Dense #to add hidden layers

# Initializing the ANN
clf = Sequential()
# Adding input layer and first hidden layer 
clf.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11)) 
# Second hidden layer
clf.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu')) 
# Adding the output layer
clf.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) 
# Compiling the ANN (applying stochastic gradient descent on the network)
clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 
# Fitting the ANN to the training set
clf.fit(X_train, y_train, batch_size = 10, epochs = 100) 


# Part 3 - Evaluating the ANN using the test set

# Predicting Test set results
y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5)

#Creating the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""
y_practice = clf.predict(scaler_x.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
y_practice = (y_practice > 0.5) """


# Part 4 - Evaluating, improving and tuning the ANN

# Evaluating
from keras.wrappers.scikit_learn import KerasClassifier # using a wrapper class of sklearn to perform k-fold cross validation
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential 
from keras.layers import Dense 
# function to build the architecture of the ANN
def build_classifier():
    clf = Sequential()
    clf.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11)) 
    clf.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu')) 
    clf.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) 
    clf.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy']) 
    return clf
# global classifier object
global_clf = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
# 10 fold cross validation
acc = cross_val_score(estimator=global_clf, X=X_train, y=y_train, cv=10, n_jobs=1)

mean_acc = acc.mean()
variance = acc.std() # high variance shows that the model may have been overfitted 
# Dropout regularization to reduce overfitting if needed


# Improving the ANN

# Parameter Tuning
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential 
from keras.layers import Dense 
def build_classifier(optimizer):
    clf = Sequential()
    clf.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11)) 
    clf.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu')) 
    clf.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) 
    clf.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy']) 
    return clf

global_clf = KerasClassifier(build_fn = build_classifier)
# hyper-parameters to optimize
parameters = {'batch_size' : [25, 32],
              'epochs': [100, 500],
              'optimizer' : ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = global_clf,
                           param_grid  = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_acc = grid_search.best_score_

