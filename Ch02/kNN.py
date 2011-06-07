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
import os
from collections import defaultdict

from numpy import array, zeros, subtract, divide, square, sqrt


def knn_classify(input_vector, training_set, labels, k):
    """Classify input vector using k nearest neighbors.

    Args:
        input_vector: The input vector to classify.
        training_set: The matrix of training examples.
        labels: The class labels.
        k: The number of neighbors to use.

    """
    diff_matrix = subtract(input_vector, training_set)
    diff_matrix_squared = square(diff_matrix)
    distances_squared = diff_matrix_squared.sum(axis=1)
    distances = sqrt(distances_squared)
    class_count = defaultdict(int)
    for i in range(k):
        smallest = distances.argmin()
        class_count[labels[smallest]] += 1
        # Rather than removing the element from the necessary arrays,
        # setting the smallest value to infinity essentially does
        # the same thing (and is much simpler and efficient).
        distances[smallest] = float('inf')

    # Find the key with the highest value by first sorting the dictionary
    # by values (highest first):
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1), reverse=True)
    # And then by returning the key associated with the highest value.
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
    dating_data, dating_labels = load_data_set('datingTestSet.txt')
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


def image_to_vector(filename):
    """Given a 32 x 32 matrix, flatten it to 1 x 1024.

    For example, if I had a 3 x 3 matrix:

        [[0,1,2],
        [3,4,5],
        [6,7,8]]

    This would return [0, 1, 2, 3, 4, 5, 6, 7 8].

    """
    vector = zeros((1, 1024))
    f = open(filename)
    for i, line in enumerate(f):
        for j in range(32):
            vector[0,32*i+j] = int(line[j])
    return vector


def handwriting_class_test():
    training_matrix, labels = _train_handwriting_data(directory='trainingDigits')

    error_count = 0.0
    total_count = 0.0
    for filename in os.listdir('testDigits'):
        expected_class_number = _get_class_number_from_filename(filename)
        vector_to_test = image_to_vector('testDigits/%s' % filename)
        classifier_result = knn_classify(vector_to_test, training_matrix,
                                         labels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % \
            (classifier_result, expected_class_number)
        if classifier_result != expected_class_number:
            error_count += 1.0
        total_count += 1.0
    print "\nthe total number of errors is: %d" % error_count
    print "\nthe total error rate is: %f" % (error_count / total_count)



def _train_handwriting_data(directory):
    labels = []
    training_files = os.listdir(directory)
    num_training_files = len(training_files)
    training_matrix = zeros((num_training_files, 1024))
    for i in range(num_training_files):
        filename = training_files[i]
        class_number = _get_class_number_from_filename(filename)
        labels.append(class_number)
        training_matrix[i,:] = image_to_vector('trainingDigits/%s' % filename)
    return training_matrix, labels


def _get_class_number_from_filename(filename):
    # The class number is indicated by the filename: "1_12.txt" -> 1
    # "2_9.txt" -> 2, "3_1.txt" -> 3, etc.
    return int(os.path.splitext(filename)[0].split('_')[0])
