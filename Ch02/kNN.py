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

from numpy import tile, array, zeros, shape, subtract, divide, square, sqrt


def knn_classify(input_vector, training_set, labels, k):
    """Classify input vector using k nearest neighbors.

    Args:
        input_vector: The input vector to classify.
        training_set: The matrix of training examples.
        labels: The class labels.
        k: The number of neighbors to use.

    """
    training_set_size = training_set.shape[0]
    diff_matrix = subtract(input_vector, training_set)
    diff_matrix_squared = square(diff_matrix)
    distances_squared = diff_matrix_squared.sum(axis=1)
    distances = sqrt(distances_squared)
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
    """Loads a dataset from a filename.

    The expected format is:

        1.0 0.9 0.8 3
        3.0 0.1 0.2 2

    Where the first three columns are the raw data, and the 4th column in the
    class label.  The data returned is a tuple of a 2d array consisting of the
    three data columns and a list containing the class labels.  For example,
    the above data would correspond to a return value of::

        array([[1.0, 0.9, 0.8],
               [3.0, 0.1, 0.2]]), [3, 2]

    """
    f = open(filename)
    class_labels = []
    data = []
    for line in f:
        columns = line.strip().split()
        data.append([float(i) for i in columns[:3]])
        class_labels.append(int(columns[-1]))
    return array(data), class_labels


def normalize(data):
    """Normalize the dataset.

    Using the equation::

        newvalue = (oldvalue - min) / (min - max)

    """
    minimum_values = data.min(0)
    maximum_values = data.max(0)
    # A vector containing the range for each column,
    # so something like [10, 14, 20] indicates that the
    # first column has a range of 10, the second a range
    # of 14, and the third column has a range of 20.
    ranges = maximum_values - minimum_values

    # Element wise computation of:
    # (old - min) / (min - max)
    normalized_data = divide(subtract(data, minimum_values), ranges)
    return normalized_data, ranges, minimum_values


def dating_class_test():
    # Hold out 10%.
    holdout_ratio = 0.10
    dating_data, dating_labels = load_data_set('datingTestSet2.txt')
    normalized_data, ranges, minimum_values = normalize(dating_data)
    num_test_vectors = int(normalized_data.shape[0] * holdout_ratio)
    error_count = 0.0
    # The ordering of the data set has no particular meaning,
    # so using the first num_test_vectors as the test vectors
    # and the remaining vectors as the training data is a
    # reasonable partitioning.
    training_data = normalized_data[num_test_vectors:,:]
    training_labels = dating_labels[num_test_vectors:]
    for i in range(num_test_vectors):
        classifier_result = knn_classify(normalized_data[i,:], training_data,
                                         training_labels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % \
            (classifier_result, dating_labels[i])
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print "the total error rate is: %f" % (
        error_count / float(num_test_vectors))
    print error_count


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
