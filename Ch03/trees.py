'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 2
@author: Peter Harrington
'''
from math import log
import operator
from pprint import pprint
from collections import defaultdict

from numpy import array, log2, multiply, divide, sum


def create_data_set():
    # Sample data set to use for testing.
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataset, labels


def calculate_entropy(dataset):
    """Calculate entropy of dataset.

    Args:
        dataset - A list of lists, where the inner lists'
            last column is the class label, for example::

                [[1, 1, 'label1'], [1, 2, 'label2']]
            See create_data_set() for example dataset.

    """
    class_labels = [row[-1] for row in dataset]
    return calculate_entropy_of_sequence(class_labels)


def calculate_entropy_of_sequence(sequence):
    """Calculate the entropy of a sequence.

    The difference between this and calculate_entropy is that
    this function expects sequence to just be single elements.

    Args:
        sequence - A list of elements, e.g [1, 1, 1, 2].

    """
    num_entries = len(sequence)
    frequency_counts = defaultdict(float)
    for element in sequence:
        frequency_counts[element] += 1
    # Scale from 0 to 1.
    distribution = divide(array(frequency_counts.values()), num_entries)
    # Each element, e_i, is defined as: e_i = p_i * log2(p_i)
    # And the entropy is the negative of the sum of the e_i's.
    return -sum(
        multiply(distribution,
                  log2(distribution))
    )


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calculate_entropy(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calculate_entropy(subDataSet)
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer


def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree


def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


def demo():
    data, labels = create_data_set()
    print "Data set:"
    for row in data:
        print row
    print "\nEntropy of original data set:", calculate_entropy(data)
    print "\nMixing up the data more:"
    data2 = data[:]
    data2[0][-1] = 'maybe'
    for row in data:
        print row
    print "\nEntropy of original data set:", calculate_entropy(data)


if __name__ == '__main__':
    demo()
