import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

clf = SVC(C=10.0,kernel="rbf")
clf.fit(features_train, labels_train)  

#### store your predictions in a list named pred
pred = clf.predict(features_test)

#from sklearn.metrics import accuracy_score
#acc = accuracy_score(pred, labels_test)

#def submitAccuracy():
#	return acc


prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
