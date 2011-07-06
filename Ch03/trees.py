'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 2
@author: Peter Harrington
'''
from cPickle import dumps, loads
import operator
from copy import deepcopy
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


def split_dataset(dataSet, axis, value):
    """Split the dataset on an axis by a certain value.

    For example, given ::

        dataset = [[1, 1, 'yes'],
                   [1, 1, 'yes'],
                   [1, 0, 'no'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
    Splitting on the 0th axis by the value 1 would return
    all elements whose 0th element equals 1.  Also The axis
    to split on is removed from the returned data.  So the
    above dataset would return::

        print split_dataset(dataset, 0, 1)
        [[1, 'yes'], [1, 'yes'], [0, 'no']]

    Args:
        dataset - A list of lists.  The inner lists must be
            the same length.
            See create_data_set() for example dataset.
        axis: The index which to look for values.
        value: The value to split on.

    """
    split_data = []
    for feature_vector in dataSet:
        if feature_vector[axis] == value:
            # These next two lines chop out the axis used for splitting.
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis+1:])
            split_data.append(reduced_feature_vector)
    return split_data


def choose_best_feature_to_split_on(dataset):
    """Find the best feature to split on.

    Returns the index that represents that feature.

    Args:
        dataset - A list of lists.  The inner lists must be
            the same length.
            See create_data_set() for example dataset.

    """
    num_features = len(dataset[0]) - 1
    base_entropy = calculate_entropy(dataset)
    best_info_gain = 0.0
    best_feature = None
    # The last column is used for the labels, so the total
    # number of columns is the length of a row in the dataset
    # minus 1.
    for i in range(len(dataset[0]) - 1):
        # Create a list of all the examples of this feature.
        current_entropy = _calculate_entropy_for_split(dataset, feature_index=i)
        # Calculate the info gain; ie reduction in entropy
        info_gain = base_entropy - current_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def _calculate_entropy_for_split(dataset, feature_index):
    unique_values = set(example[feature_index] for example in dataset)
    current_entropy = 0.0
    for value in unique_values:
        sub_dataset = split_dataset(dataset, feature_index, value)
        prob = len(sub_dataset) / float(len(dataset))
        current_entropy += prob * calculate_entropy(sub_dataset)
    return current_entropy


def majority_count(classes):
    class_count = defaultdict(float)
    for vote in classes:
        class_count[vote] += 1
    sortedClassCount = sorted(class_count.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def create_tree(dataset, labels):
    """Construct a decision tree.

    Args:
        dataset - A list of lists, where the inner lists'
            last column is the class label, for example::

                [[1, 1, 'label1'], [1, 2, 'label2']]
            See create_data_set() for example dataset.
        labels - A list of textual descriptions of the
            features.  For example, in the list above for
            dataset, the corresponding labels might be
            'no surfacing' and 'flippers' which correspond
            to what the two columns mean (the third column
            is the class).

    Returns the constructed decision tree, which is represented
    as a nested dictionary.  There's a single key in the top
    level dictionary which represents the root node. The
    subsequent dictionaries represent various features
    and their possible values.  A leaf node represents
    the class label.  For example::

        {'no surfacing':
            {0: 'no',
             1: {'flippers':
                 {0: 'no',
                  1: 'yes'}}}}

    """
    labels = labels[:]
    class_labels = [example[-1] for example in dataset]
    # Stop splitting when all of the classes are equal.
    if len(set(class_labels)) == 1:
        return class_labels[0]
    # Stop splitting when there are no more features in dataset.
    if len(dataset[0]) == 1:
        return majority_count(class_labels)
    best_feature_index = choose_best_feature_to_split_on(dataset)
    best_feature_name = labels[best_feature_index]
    tree = {best_feature_name: {}}
    del labels[best_feature_index]
    for value in set(el[best_feature_index] for el in dataset):
        tree[best_feature_name][value] = create_tree(
            split_dataset(dataset, best_feature_index, value),
            labels)
    return tree


def classify(tree, labels, input_vector):
    root = tree.keys()[0]
    children = tree[root]
    feature_index = labels.index(root)
    key = input_vector[feature_index]
    new_root = children[key]
    if isinstance(new_root, dict):
        class_label = classify(new_root, labels, input_vector)
    else:
        class_label = new_root
    return class_label


def store_tree(input_tree,filename):
    f = open(filename,'w')
    dumps(input_tree, f)
    f.close()


def grab_tree(filename):
    f = open(filename)
    return pickle.load(f)


def demo():
    data, labels = create_data_set()
    print "Data set:"
    for row in data:
        print row
    print "\nEntropy of original data set:", calculate_entropy(data)
    print "\nMixing up the data more:"
    data2 = deepcopy(data)
    data2[0][-1] = 'maybe'
    for row in data2:
        print row
    print "\nEntropy of original data set:", calculate_entropy(data2)

    print "Data set:"
    for row in data:
        print row
    print "\n\nSplitting data on 0th axis by value 1:"
    print split_dataset(data, 0, 1)
    print "\n\nThe best feature to split on:"
    print choose_best_feature_to_split_on(data)

    print "Creating decision tree:"
    tree = create_tree(data, labels)
    print tree
    print "\nClassifying, no surfacing=true, flippers=true"
    print "    --> ", classify(tree, labels, [1, 1])


if __name__ == '__main__':
    demo()
