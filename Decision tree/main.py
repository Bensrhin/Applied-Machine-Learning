#!/usr/bin/env python3
import ToyData as td
import ID3
from sklearn import datasets
import numpy as np
from sklearn import tree, metrics, datasets
from sklearn.model_selection import train_test_split
import collections

from sklearn.metrics import classification_report, confusion_matrix

def main():

    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()
    print(attributes, "\n cl: ", classes, "\n data: ", data, "\n targ: ", target, "\n dt2: ", data2, "\n tar2: ", target2)

    id3 = ID3.ID3DecisionTreeClassifier()
    
#     print(id3.find_split_attr(data, target, attributes, classes))
    myTree = id3.fit(data, target, attributes, classes)
#     print(myTree)
    plot = id3.make_dot_data()
    plot.render("testTree")
    predicted = id3.predict(data2, myTree)
    print(predicted)
    
    
    digits = datasets.load_digits()
    # Features
    x = digits.data
    # Labels
    y = digits.target
    # images
    images = digits.images
    x_train, x_test, y_train, y_test, img_tr, img_ts = train_test_split(x, y, images, test_size=0.3)
    attributes = collections.OrderedDict()
    for i in range(64):
        attributes[i] = [j for j in range(17)]
    t = digits.target_names
    target_names=np.array(t).astype('str').tolist()
    
    clf = ID3.ID3DecisionTreeClassifier()
    tree = clf.fit(x_train, y_train, attributes, target_names)
    y_hat = clf.predict(x_test, tree)
    print(confusion_matrix(y_test, y_hat))
    print(classification_report(y_test, y_hat))
    X = np.array(x, dtype='<U21')
    for i in range(len(x)):
        for j in range(64):
            if x[i,j] < 5:
                X[i,j] = 'dark'
            elif x[i,j] > 10:
                X[i,j] = 'light'
            else:
                X[i,j] = 'gray'
    
    x_train, x_test, y_train, y_test, img_tr, img_ts = train_test_split(X, y, images, test_size=0.3)
    attributes = collections.OrderedDict()
    for i in range(64):
        attributes[i] = ['dark', 'light', 'gray']
    clf = ID3.ID3DecisionTreeClassifier()
    tree = clf.fit(x_train, y_train, attributes, target_names)
    y_hat = clf.predict(x_test, tree)
    print(confusion_matrix(y_test, y_hat))
    print(classification_report(y_test, y_hat))
if __name__ == "__main__": main()
