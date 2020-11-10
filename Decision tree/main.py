#!/usr/bin/env python3

import numpy as np
import collections
import ToyData as td
import ID3
from sklearn import datasets
from sklearn import tree, metrics, datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split



def main():
    
    #----------------------- ToyData -------------------------
    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()
    # ID3 classifier
    id3 = ID3.ID3DecisionTreeClassifier()
    
    # Fitting the model
    myTree = id3.fit(data, target, attributes, classes)
#     print(myTree)

    # Visualization of the Tree
    plot = id3.make_dot_data()
    # Text form
    print("Tree:\n", plot.source, "\n")
    # graphic form
    plot.render("testTree")
    
    # Testing the model
    predicted = id3.predict(data2, myTree)
    
    # Evaluation
    print("Confusion matrix:\n", confusion_matrix(predicted, target2), "\n")
    print("classification report:\n", classification_report(predicted, target2), "\n")
    
    
    # ------------------------ Digits -----------------------------------
    
    digits = datasets.load_digits()
    x = digits.data
    y = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    
    # Ordered Attributes : 1..16
    attributes = collections.OrderedDict()
    for i in range(64):
        attributes[i] = [j for j in range(17)]
    
    # Converting classes to str
    t = digits.target_names
    target_names=np.array(t).astype('str').tolist()
    
    # The classifier ID3
    clf = ID3.ID3DecisionTreeClassifier()
    
    # Fitting the model
    tree = clf.fit(x_train, y_train, attributes, target_names)
    
    # Visualization
    plot = clf.make_dot_data()
    plot.render("digitsTree_16")
    # print("Tree:\n", plot.source, "\n")
    
    
    # Testing the model
    y_hat = clf.predict(x_test, tree)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_hat), "\n")
    print("classification report:\n", classification_report(y_test, y_hat), "\n")
    
    
    # Modify your data set to contain only three values for the attributes
    X = np.array(x, dtype='<U21')
    for i in range(len(x)):
        for j in range(64):
            if x[i,j] < 5:
                X[i,j] = 'dark'
            elif x[i,j] > 10:
                X[i,j] = 'light'
            else:
                X[i,j] = 'gray'
    # Splitting the new data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Ordered Attributes : dark, light, gray
    attributes = collections.OrderedDict()
    for i in range(64):
        attributes[i] = ['dark', 'light', 'gray']
        
    # building a new model
    clf = ID3.ID3DecisionTreeClassifier()
    tree = clf.fit(x_train, y_train, attributes, target_names)
    
    # Visualize
    plot = clf.make_dot_data()
    plot.render("digitsTree_3")
    
    # Testing it
    y_hat = clf.predict(x_test, tree)
    
    # Evaluation:
    print("Confusion matrix:\n", confusion_matrix(y_test, y_hat), "\n")
    print("classification report:\n", classification_report(y_test, y_hat), "\n")
    
    
if __name__ == "__main__": main()
