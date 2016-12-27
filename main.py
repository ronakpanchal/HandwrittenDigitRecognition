import pickle
import random
import numpy
from os import listdir
from os.path import join
import os
from PIL import Image


no_of_labels = 10
no_of_features = 784
no_of_neurons = 1000
learning_rate = 0.001
reg_lambda = 0.01


# Logistic Regression implementation
def logistic_regression(train_data, train_label):
    weights = numpy.random.random((no_of_features+1, no_of_labels))

    for epochs in range(0, 3):
        for i in range(0, len(train_data)):
            pr_arr = calculate_softmax(weights, train_data[i])
            target_vector = numpy.zeros(shape=no_of_labels)
            target_vector[int(train_label[i])] = 1
            _entropy_error = cross_entropy_error(pr_arr, target_vector)
            diff = pr_arr - target_vector
            x = numpy.asmatrix(train_data[i])
            grad = numpy.dot(numpy.transpose(x), numpy.asmatrix(diff))
            weights = weights - (learning_rate * grad)
    return weights, _entropy_error


def shuffle_dataset(dataset):
    temp_array = numpy.column_stack((dataset[0], dataset[1]))
    dataset = temp_array
    random.shuffle(dataset)
    return dataset[:, :no_of_features], dataset[:, no_of_features]


def cross_entropy_error(probabilities, target_vector):
    error_sum = -numpy.sum(numpy.log(probabilities) * target_vector)
    return error_sum / no_of_labels


def calculate_softmax(_weights, x):
    pr = numpy.zeros(shape=no_of_labels)
    w_x = numpy.dot(x, _weights) + numpy.ones(no_of_labels)
    softmax_mat = numpy.array(numpy.exp(w_x)).ravel()
    sum_matrix = numpy.sum(softmax_mat)
    for index in range(0, 10):
        pr[index] = softmax_mat[index] / sum_matrix
    return pr


def lr_classification_rate(w, dataset, labels):
    no_of_mismatch = 0
    for index in range(0, len(dataset)):
        probabilities = calculate_softmax(w, dataset[index])
        ind = numpy.argmax(probabilities)
        if ind != int(labels[index]):
            no_of_mismatch += 1
    return no_of_mismatch/len(dataset), no_of_mismatch


def classification_rate_ann(data, labels, _wj, _wk):
    mismatch = 0
    for l in range(0, len(data)):
        x_i = data[l]
        _zj = numpy.dot(x_i, _wj)
        _a1 = sigmoid(_zj)
        _ak = numpy.dot(_a1, _wk)
        _ak = numpy.exp(_ak)

        sum_exp = numpy.sum(_ak)
        pr = _ak / sum_exp
        ind = numpy.argmax(pr)
        error_rate = cross_entropy_error(pr, pr)
        if ind != int(labels[l]):
            mismatch += 1
    return mismatch/len(data), error_rate, mismatch


def calculate_dataset_error(dataset, labels, weights):
    sum_of_errors = 0
    for i in range(0, len(dataset)):
        pr_arr = calculate_softmax(weights, dataset[i])
        target_vector = numpy.zeros(shape=no_of_labels)
        target_vector[int(labels[i])] = 1
        sum_of_errors += cross_entropy_error(pr_arr, target_vector)
    mean_error = sum_of_errors / (len(dataset))
    return mean_error


def sigmoid(a):
    b = 1 + numpy.exp(-a)
    return 1/b


def rectified_linear(n):
    return numpy.log(1+numpy.exp(n))


def sigmoid_derivative(m):
    k = 1 - sigmoid(m)
    return sigmoid(m) * k


def simple_neural_network(_inputs, _labels):

    w_j = numpy.random.random((no_of_features+1, no_of_neurons))/numpy.sqrt(no_of_features)
    w_k = numpy.random.random((no_of_neurons, no_of_labels))/numpy.sqrt(no_of_neurons)
    bias_1 = numpy.ones((no_of_features+1, 1))
    bias_2 = numpy.ones((1, no_of_labels))
    w_j = numpy.c_[w_j, bias_1]
    w_k = numpy.vstack((w_k, bias_2))
    eta = 0.001  # learning rate

    for epochs in range(0, 7):  # considering number of epochs equal to 6
        for index in range(0, len(_inputs)):  # the number of passes is 50,000 ,batch size is 1
            _input = _inputs[index]
            aj = numpy.dot(_input, w_j)
            zj = sigmoid(aj)
            ak = numpy.dot(zj, w_k)
            ak = numpy.exp(ak)
            _sum_exp = numpy.sum(ak)

            probabilities = ak / _sum_exp
            probabilities = numpy.array(probabilities).ravel()

            labels = numpy.zeros(shape=no_of_labels)
            labels[int(_labels[index])] = 1
            _error = cross_entropy_error(probabilities, labels)

            dk = probabilities - labels
            del_k = numpy.transpose(numpy.asmatrix(zj)) * dk
            x1 = numpy.dot(w_k, numpy.transpose(dk))
            temp = numpy.asmatrix(sigmoid_derivative(aj) * x1)  # changed from zj to aj
            del_j = numpy.asmatrix(_input).T * temp
            w_j -= (eta * del_j)
            w_k -= (eta * del_k)
    print(' The Single neural network cross entropy error for training set is ', _error * 100)
    return w_j, w_k


with open('mnist.pkl', 'rb') as f:
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    training_data, training_labels = shuffle_dataset(training_data)
    validation_data, validation_labels = shuffle_dataset(validation_data)
    test_data, testing_labels = shuffle_dataset(test_data)

    training_data = numpy.c_[numpy.ones((len(training_data), 1)), training_data]
    validation_data = numpy.c_[numpy.ones((len(validation_data), 1)), validation_data]
    test_data = numpy.c_[numpy.ones((len(test_data), 1)), test_data]

    dirs = os.listdir('Numerals')
    images = list()
    counter = 0
    X = numpy.zeros((19999, 784))
    Y = numpy.zeros(19999)
    for file in dirs:
        for image in listdir(join('Numerals', file)):
            name, ext = os.path.splitext(image)
            if ext == '.png':
                filename = join(join('Numerals', file), image)
                im = Image.open(filename)
                im = im.resize((28, 28))
                im = im/numpy.sum(im)
                X[counter] = im.flatten()
                Y[counter] = file
                counter += 1
    X = numpy.c_[numpy.ones((len(X), 1)), X]

    weight_mat, entropy_error = logistic_regression(training_data, training_labels)
    print('The cross entropy error rate  for training set using logistic regression is ', entropy_error * 100)

    print('########################################################################')
    print('Following values are calculated using logistic regression')

    mean_error_validation = calculate_dataset_error(validation_data, validation_labels, weight_mat)
    print('The cross entropy error rate for validation set is ', mean_error_validation * 100)

    mean_error_testing = calculate_dataset_error(test_data, testing_labels, weight_mat)
    print('The cross entropy error rate for testing set is ', mean_error_testing * 100)

    mean_error_testing = calculate_dataset_error(X, Y, weight_mat)
    print('The cross entropy error rate for USPS data set is ', mean_error_testing * 100)

    correct_predictions, no_mismatch = lr_classification_rate(weight_mat, training_data, training_labels)
    print('Classification error rate  for training set is ', correct_predictions)
    print('No of mismatch using LR for training set is ', no_mismatch)

    correct_predictions, no_mismatch = lr_classification_rate(weight_mat, validation_data, validation_labels)
    print('Classification error rate  for validation set is ', correct_predictions)
    print('No of mismatch using LR for validation set is ', no_mismatch)

    correct_predictions, no_mismatch = lr_classification_rate(weight_mat, test_data, testing_labels)
    print('Classification error rate  for testing set is ', correct_predictions)
    print('No of mismatch using LR for testing set is ', no_mismatch)

    correct_predictions, no_mismatch = lr_classification_rate(weight_mat, X, Y)
    print('Classification error rate  for USPS data set is ', correct_predictions)
    print('No of mismatch using LR for USPS data set is ', no_mismatch)

    print('########################################################################')
    print('Following values are calculated using single hidden layer neural network')
    # call simple neural network subroutine
    wj, wk = simple_neural_network(training_data, training_labels)
    # calculate argmax value for validation set
    no_of_mismatches, err, no_mismatch = classification_rate_ann(training_data, training_labels, wj, wk)
    print('Classification error rate for training  data is equal to ', no_of_mismatches, 'and cross entropy error is ', err)
    print('The number of mismatches for training set is ', no_mismatch)

    no_of_mismatches, err, no_mismatch = classification_rate_ann(validation_data, validation_labels, wj, wk)
    print('Classification error rate for validation  data is equal to ', no_of_mismatches, 'and cross entropy error is ', err)
    print('The number of mismatches for validation set is ', no_mismatch)

    no_of_mismatches, err, no_mismatch = classification_rate_ann(test_data, testing_labels, wj, wk)
    print('Classification error rate for testing  data  is equal to ', no_of_mismatches, 'and cross entropy error is ', err)
    print('The number of mismatches for testing  set is ', no_mismatch)

    no_of_mismatches, err, no_mismatch = classification_rate_ann(X, Y, wj, wk)
    print('Classification error rate for USPS data set is is equal to ', no_of_mismatches, 'and cross entropy error is ', err)
    print('The number of mismatches for USPS data  set is ', no_mismatch)





        



