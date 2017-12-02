import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('mushrooms.csv')
dataset = dataset.drop(['veil-type','veil-color','gill-attachment','stalk-root'],axis=1)
X = dataset.iloc[:,1:]
y = dataset.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X = X.apply(labelencoder_X.fit_transform) # encoding multiple columns
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y,)

from sklearn.svm import SVC
classifier = SVC(kernel='linear')


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
