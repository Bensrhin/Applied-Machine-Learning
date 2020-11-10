from collections import Counter
from graphviz import Digraph
from math import log
import numpy as np

#-----------------------------------------------------------------------------
# Functions:
#-----------------------------------------------------------------------------

def get_labels(attribute_column):
    labels = {}
    for val in attribute_column:
        if val not in labels.keys():
            labels[val] = 1
        else:
            labels[val] += 1
    return labels

def entropy(labels, length):
    I = 0
    for label in labels.keys():
        p = labels[label] / length
        I -= p * log(p, 2)
    return I

def new_data(data, target, attribute_values, idx_attr):
    datas = {}
    targets = {}
    for values in attribute_values:
        targets[values] = target[values == data[:, idx_attr]]
        if len(targets[values]) != 0:
            datas[values] = np.delete(data[values == data[:, idx_attr]], idx_attr, 1)
        else:
            datas[values] = data[values == data[:, idx_attr]]
    return datas, targets

def index_attribute(attributes, attribute_key):
    idx = 0
    for key in attributes.keys():
        if attribute_key == key:
            return idx
        idx += 1
    return None

def most_common_class(target):
    classes = {}
    for val in target:
        if val not in classes:
            classes[val] = 1
        else:
            classes[val] += 1
    return max(classes.keys(), key=(lambda key: classes[key]))

#-----------------------------------------------------------------------------
# Classifier
#-----------------------------------------------------------------------------
class ID3DecisionTreeClassifier :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree: ID3')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit


    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'value': None, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': None, 'attributes' : None}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes') and (k != 'attributes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])
        #print(nodeString)
        
        return


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot

    
    
    
    def find_split_attr(self, data, target, attributes, classes):
        data = np.array(data)
        target = np.array(target)
        labels = get_labels(target)
        
        # I(S)
        n = len(target)
        I = entropy(labels, n)
        Gain = {}
        for attr in attributes:
            idx_attr = index_attribute(attributes, attr)
            datas, targets = new_data(data, target, attributes[attr], idx_attr)
            average_entropies = 0
            for key in datas.keys():
                labels = get_labels(targets[key])
                nv = len(targets[key])
                I_v = entropy(labels, nv)
                average_entropies += (nv * I_v) / n
            gain = I - average_entropies
            Gain[attr] = gain
        
        return max(Gain.keys(), key=(lambda key: Gain[key])), I
    

    def fit(self, data, target, attributes, classes, isRoot=True, value='-', iD=-1):
        data = np.array(data)
        target = np.array(target)
        root = self.new_ID3_node()
        
        labels = get_labels(target)
        n = len(target)
        I = entropy(labels, n)
        root['samples'] = n
        root['value'] = value
        
        
        if isRoot:
            root['attributes'] = dict(attributes)
            root.update({'value': '-'})
        root['classCounts'] = labels
        
        
        # If all samples belong to one class
        if len(set(target)) == 1:
            root['label'] = most_common_class(target)
            root['entropy'] = 0
            self.add_node_to_graph(root, iD)
        # If attributes is empty
        elif attributes == {}:
            root['label'] = most_common_class(target)
            root['entropy'] = I

            self.add_node_to_graph(root, iD)
        else:
            A, I = self.find_split_attr(data, target, attributes, classes)
            root['attribute'] = A
            root['entropy'] = I
            
            idx_attr = index_attribute(attributes, A)
            datas, targets = new_data(data, target, attributes[A], idx_attr)
            root['nodes'] = {}
            self.add_node_to_graph(root, iD)
            for key in datas.keys():

                # If Samples(v) is empty      
                if targets[key].size==0:
                    branch= self.new_ID3_node()
                    branch['samples'] = 0
                    root['classCounts'] = {}
                    branch.update({'value': key})
                    branch.update({'label': most_common_class(target)})
                    root['nodes'][key] = branch
                    self.add_node_to_graph(branch, root['id'])
                else:
                    
                    attributes_copy = dict(attributes)
                    attributes_copy.pop(A, None)
                    node = self.fit(datas[key], targets[key], attributes_copy, classes, False, key, root['id'])
                    root['nodes'][key] = node
                    
        return root


    def predicted_rek(self, node, x, attributes):
        
        if node['nodes']==None or node['nodes']=={}:
            return node['label']
        else:
            idx_attr = index_attribute(attributes, node['attribute'])
            c = node['nodes'][x[idx_attr]]
            return self.predicted_rek(c, x, attributes)
                
                
    def predict(self, data, tree) :
        attributes = tree['attributes']
        data = np.array(data)
        predicted = list()
        for i in range (data.shape[0]):
            x = data[i,:]
            predicted += [self.predicted_rek(tree, x, attributes)]
        
        return predicted