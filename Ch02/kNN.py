"""
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:   inX: vector to compare to existing dataset (1xN)
         dataSet: size m data set of known vectors (NxM)
         labels: data set labels (1xM vector)
         k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin
"""
import operator
from os import listdir

from numpy import tile, array, zeros, shape


def knn_classify(input_vector, training_set, labels, k):
    """Classify input vector using k nearest neighbors.

    Args:
        input_vector: The input vector to classify.
        training_set: The matrix of training examples.
        labels: The class labels.
        k: The number of neighbors t ouse.

    """
    training_set_size = training_set.shape[0]
    diff_matrix = tile(input_vector, (training_set_size, 1)) - training_set
    diff_matrix_squared = diff_matrix ** 2
    distances_squared = diff_matrix_squared.sum(axis=1)
    distances = distances_squared ** 0.5
    sorted_distance_indices = distances.argsort()
    class_count = {}
    for i in range(k):
        current_label = labels[sorted_distance_indices[i]]
        class_count[current_label] = class_count.get(current_label, 0) + 1

    sorted_class_count = sorted(class_count.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def load_data_set(filename):
    f = open(filename)
    class_labels = []
    data = []
    for line in f:
        columns = line.strip().split()
        data.append([float(i) for i in columns[:3]])
        class_labels.append(int(columns[-1]))
    return array(data), class_labels


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    # Element wise division
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def dating_class_test():
    # Hold out 10%.
    hoRatio = 0.50
    datingDataMat, datingLabels = load_data_set('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = knn_classify(normMat[i,:], normMat[numTestVecs:m,:],
                                        datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % \
            (classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))
    print errorCount


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = knn_classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % \
            (classifierResult, classNumStr)
        if classifierResult != classNumStr:
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))
