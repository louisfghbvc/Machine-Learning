import gzip
import numpy as np
import struct
from random import seed
import random
import math
import matplotlib.pyplot as plt

def ReadImageData(s):
    ''' 
    Introduction:
        Read data image according to mnist dataset
    Argument:
        s -- string of the file path
    Return:
        data -- the input data shape of (number of images, 784)
    '''
    with gzip.open(s, 'r') as f:
        f.read(4)                                                    # no use value
        num_images = int.from_bytes(f.read(4), byteorder='big')      # big-endian
        f.read(8)                                                    # just skip 8 byte, row and col
        img_size = 28                                                # always fix
        buf = f.read(img_size * img_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, img_size*img_size)
        data /= 255.
    return data

def ReadImageLabel(s):
    ''' 
    Introduction:
        Read data label according to mnist dataset
    Argument:
        s -- string of the file path
    Return:
        data -- the input data shape of (number of labels,)
    '''
    with gzip.open(s, 'r') as f:
        f.read(4)                                                    # no use value
        num_images = int.from_bytes(f.read(4), byteorder='big')      # big-endian
        buf = f.read(num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    return data

def SuffleXY(x, y):
    '''
    Introduction:
        Suffle data, make good result
    Argument:
        x -- origin input data of shape (number of image, 784)
        y -- origin output label of shape (number of image,)
    Return:
        _x -- input data of shape (number of image, 784)
        _y -- output label of shape (number of image,)
    '''
    suffle_ind = np.random.permutation(60000)
    _x = []
    _y = []
    for i in range(60000):
        _x.append(x[suffle_ind[i]])
        _y.append(y[suffle_ind[i]])
    return np.array(_x), np.array(_y)

def oneHotEncode(label):
    '''
    Introduction:
        Make label output 1 to 10
    Argument:
        label -- tag of the image shape(nums of label,)
    Return:
        res -- tag of image, shape is (10, nums of label)
    '''
    res = np.zeros((label.size, 10))
    for idx, ele in enumerate(res):
        ele[int(label[idx])] = 1.
    return res.T

def Softmax(x):
    '''
    Introduction:
        The activation func make input sum probability to 1
        Use exponent method to do above
        Use normalization to avoid overflow issue
        Sum value axis = 0, because x of shape(10, batch_size)
    Argument:
        x -- the output data of shape (10, num of label)
    Returns:
        expX -- probability data of shape (10, nums of label)
    '''
    x = x - np.max(x)
    expX = np.exp(x)
    return expX / np.sum(expX, axis = 0)

def ReLU(x):
    '''
    Introduction:
        A function make all array value, if x > 0, y > 0, else y = 0 
    Argument:
        x -- any shape of numpy array
    Returns:
        numpy -- any shape of numpy array 
    '''
    return np.maximum(0, x)

def CrossEncropy(t, y):    
    '''
    Introduction:
        The cost equal to sum of each val t*logy
    Argument:
        t -- tag of the target image label(10, nums of label)
        y -- tag of the neuron image label(10, nums of label)
    Returns:
        val -- sum of loss value 
    '''
    return -np.sum(t * np.log(y + 1e-7))

def Predict(x, param):
    '''
    Introduction:
        Give input data, get the output after neuron forward. The first layer
        activation func is ReLU, and second layer activation func is Softmax.
    Argument:
        x -- input data of shape (784, nums of data)
        param -- the information of all neuron param data
    Returns:
        val -- output data after forward, shape (10, nums of data) 
    '''
    Z1 = np.dot(param["W1"], x) + param["b1"]
    A1 = ReLU(Z1)
    Z2 = np.dot(param["W2"], A1) + param["b2"]
    A2 = Softmax(Z2)
    return A2

def CalAcc(y, target):
    '''
    Introduction:
        Give model predict label and target label,
        calculate the accuracy
    Argument:
        y -- model predict label (10, nums of label)
        target -- correct label (10, nums of label)
    Returns:
        acc -- percent of equal labels
    '''
    y = y.T.copy()
    target = target.T.copy()
    return sum(np.argmax(y[idx]) == np.argmax(target[idx]) for idx in range(len(y))) / len(y) * 100

def InitParam(n_input, n_hidden, n_output):
    '''
    Introduction:
        Initialize all neuron param, random all weight use normal distribution, 
        and divide a sqrt value make the param distinct.
    Argument:
        n_input -- size of the input layer
        n_hidden -- size of the hidden layer
        n_output -- size of the output layer
    Returns:
        python dictionary containing param:
            W1 -- weight matrix of shape (n_hidden, n_input)
            b1 -- bias vector of shape (n_hidden, 1)
            W2 -- weight matrix of shape (n_output, n_hidden)
            b2 -- bias vector of shape (n_output, 1)
    '''
    W1 = np.random.randn(n_hidden, n_input) * np.sqrt(1. / n_input)
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(n_output, n_hidden) * np.sqrt(1. / n_hidden)
    b2 = np.zeros((n_output, 1))
    
    param = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return param

def Forward(x, param):
    '''
    Introduction:
        Give input data, get the output after neuron forward. The first layer
        activation func is ReLU, and second layer activation func is Softmax.
    Argument:
        x -- input data of size (n_input, batch_size)
        param -- python dictionary with param
    Returns:
        A2 -- The softmax output of the second activation shape is (n_output, batch_size)
        cache -- python dictionary with "Z1", "A1", "Z2" and "A2"
    '''
    Z1 = np.dot(param["W1"], x) + param["b1"]
    A1 = ReLU(Z1)
    Z2 = np.dot(param["W2"], A1) + param["b2"]
    A2 = Softmax(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def Backward(param, cache, x, y):
    '''
    Introduction:
        Implement the backward propagation.
    Arguments:
        param -- python dictionary with param
        cache -- python dictionary with "Z1", "A1", "Z2" and "A2"
        x -- input data of shape (784, batch_size)
        y -- target label of shape (10, batch_size)
    Returns:
        grads -- python dictionary with gradients 
    '''
    batch_size = x.shape[1]
    W2 = param["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]

    # Derivative Softmax + CrossEntropy
    dZ2 = A2 - y

    # Output Layer Derivative w, b
    dW2 = (1./batch_size) * np.dot(dZ2, A1.T)
    # Shape of db2 (10, 1)
    db2 = (1./batch_size) * np.sum(dZ2, axis=1, keepdims=True)

    # Chain rule
    dA1 = np.dot(W2.T, dZ2)
    # Derivative ReLU
    dZ1 = np.array(dA1).copy()
    # Check Z1 value
    dZ1[Z1 <= 0] = 0
    
    # Hidden layer Derivative w, b
    dW1 = (1./batch_size) * np.dot(dZ1, x.T)
    # Shape of db1 (n_hidden, 1)
    db1 = (1./batch_size) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

def UpdateParam(param, grads, learning_rate = 0.1):
    '''
    Introduction:
        Updates param using the gradient descent
    Arguments:
        param -- python dictionary contain param
        grads -- python dictionary contain gradients 
        learning_rate -- the step of the change 
    Returns:
        param -- python dictionary param
    '''
    param["W1"] = param["W1"] - learning_rate * grads["dW1"]
    param["b1"] = param["b1"] - learning_rate * grads["db1"]
    param["W2"] = param["W2"] - learning_rate * grads["dW2"]
    param["b2"] = param["b2"] - learning_rate * grads["db2"]
    return param

def TrainModel(x, y, x_valid, y_valid, n_hidden, mini_batch, max_epoch, learning_rate):
    '''
    Introduction:
        2-layer neuron network
        Train implement, no early stopping
        Stop until max_epoch
    Arguments:
        x -- data of shape (784, nums of data)
        y -- label of shape (10, nums of label)
        x_valid -- validation of the input dataset (784, nums of data)
        y_valid -- validation of the output dataset (10, nums of label)
        n_hidden -- size of the hidden layer
        mini_batch -- batch_size of each train
        max_epoch -- stop param
        learning_rate -- step of learning
    Returns:
        param -- final param learn by the model
    '''
    n_input = x.shape[0]
    n_output = y.shape[0]

    # Data visualize
    plot_x = [i for i in range(max_epoch)]
    plot_train_loss = []
    plot_valid_loss = []
    
    # Initialize 
    param = InitParam(n_input, n_hidden, n_output)

    for epoch in range(0, max_epoch):

        # each mini-batch
        for i in range(0, x.shape[1]//mini_batch):
            # Get batch
            x_batch = x[:, 0+i*mini_batch : mini_batch+i*mini_batch]
            y_batch = y[:, 0+i*mini_batch : mini_batch+i*mini_batch]

            # Forward, Backward, Update
            _, cache = Forward(x_batch, param)
            grads = Backward(param, cache, x_batch, y_batch)
            param = UpdateParam(param, grads, learning_rate)

        # Observation predict data
        train_p = Predict(x, param)
        valid_p = Predict(x_valid, param)
        train_loss = CrossEncropy(y, train_p) / x.shape[1]
        valid_loss = CrossEncropy(y_valid, valid_p) / x_valid.shape[1]
        print('epoch: {}, train_loss: {:.6f}, validation_loss: {:.6f}, val_acc: {:.4f}'.format(epoch, train_loss, valid_loss, CalAcc(valid_p, y_valid)))
        plot_train_loss.append(train_loss)
        plot_valid_loss.append(valid_loss)

    # plot loss graph
    plt.plot(plot_x, plot_train_loss, lw=2, label='train_loss')
    plt.plot(plot_x, plot_valid_loss, lw=2, label='valid_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Show loss')
    plt.show()

    return param

if __name__ == "__main__":
    X = ReadImageData('./MNIST/train-images-idx3-ubyte.gz')
    Y = ReadImageLabel('./MNIST/train-labels-idx1-ubyte.gz')
    X_test = ReadImageData('./MNIST/t10k-images-idx3-ubyte.gz')
    Y_test = ReadImageLabel('./MNIST/t10k-labels-idx1-ubyte.gz')

    # Suffle data 
    X, Y = SuffleXY(X, Y)

    # transfrom
    Y = oneHotEncode(Y)

    # split data
    d = 42000
    X_train, X_valid = X[:d].T, X[d:].T
    Y_train, Y_valid = Y[:, :d], Y[:, d:]

    # start training model
    param = TrainModel(X_train, Y_train, X_valid, Y_valid, n_hidden = 500, mini_batch = 60, max_epoch = 15, learning_rate = 0.1)

    # Predict test data
    Y_test = oneHotEncode(Y_test)
    X_test = X_test.T
    p = Predict(X_test, param)
    print('test acc: {:.4f}%'.format(CalAcc(p, Y_test)))