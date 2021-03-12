import numpy as np
import matplotlib.pyplot as plt
import os

def Softmax(x):
    '''
    Introduction:
        The activation func make input sum probability to 1
        Use exponent method to do above
        Use normalization to avoid overflow issue
        Sum value axis = 0, because x of shape(classified labal nums, nums of data)
    Argument:
        x -- the output data of shape (classified labal nums, nums of data)
    Returns:
        expX -- probability data of shape (classified labal nums, nums of data)
    '''
    x = x - np.max(x, axis = 0)
    expX = np.exp(x)
    return expX / np.sum(expX, axis = 0)

class ReLU:
    def __call__(self, x):
        '''
        Introduction:
            Overloading operator(). to do function like method.
            make value >= 0.
        Argument:
            x -- the input data of any shape
        Returns:
            ndarray -- same shape as input data
        '''
        return np.maximum(0, x)
    
    def backward(self, da, x):
        '''
        Introduction:
            calculate activation gradient of input data x
            make da < 0 array value = 0
        Argument:
            x -- 2D ndarray. input data 
            da -- 2D ndarray. gradient come from next layer
        Returns:
            dz -- 2D ndarray. same shape as da
        '''
        dz = da.copy()
        dz[x <= 0] = 0
        return dz

def CrossEncropy(t, y):    
    '''
    Introduction:
        The cost equal to sum of each val t*logy
    Argument:
        t -- tag of the target image label(classified labal nums, nums of data)
        y -- tag of the neuron image label(classified labal nums, nums of data)
    Returns:
        float -- sum of loss value 
    '''
    return -np.sum(t * np.log(y + 1e-7))

def CalAcc(y, target):
    '''
    Introduction:
        Give model predict label and target label,
        calculate the accuracy. 
        Iterate data to check the maximum value index of each is same or not.
    Argument:
        y -- model predict label (classified labal nums, nums of label dataset)
        target -- correct label (classified labal nums, nums of label dataset)
    Returns:
        float -- percent of equal labels
    '''
    y = y.T
    target = target.T
    return sum(np.argmax(y[idx]) == np.argmax(target[idx]) for idx in range(len(y))) / len(y) * 100

def SuffleXY(data, label):
    '''
    Introduction:
        SuffleXY same time. the data and label shape[0] must be same.
        Use permutation index to do random suffle.
    Argument:
        data -- input data. that shape (nums, ...)
        label -- output data that shape (nums, ...)
    Returns:
        data -- after suffle data
        label -- after suffle label
    '''
    r_idx = np.random.permutation(len(label))
    data = data[r_idx]
    label = label[r_idx]
    return data, label

def OneHotEncode(s):
    '''
    Introduction:
        Encode string label. Just lazy dictionary.
        Make string to a vector.
    Argument:
        s -- string of label name
    Returns:
        list -- input label after encode
    '''
    return {'Carambula': [1, 0, 0], 'Lychee': [0, 1, 0], 'Pear': [0, 0, 1]}[s]

def ReadData(isTrain, labelList):
    '''
    Introduction:
        Read dataset. Find the folder in the labelList.
        List them and join.
    Argument:
        isTrain -- boolean value, determine train or test
        labelList -- label name, must be in train or test folder
    Returns:
        data -- 4D ndarray data e.g (nums, 32, 32, 3)
        label -- 2D ndarray label e.g (nums, encode)
    '''
    s = 'train' if isTrain else 'test'
    data = []
    label = []

    for labelName in labelList:
        path = './Data_' + s + '/' + labelName
        fileList = os.listdir(path)

        # read only 3 channels
        for fileName in fileList:
            data.append(plt.imread(path + '/' + fileName)[:,:,:3])
            label.append(OneHotEncode(labelName))

    return np.array(data), np.array(label)

def SplitData(x, y, p = 0.7):
    '''
    Introduction:
        Split data into train and validation.
        Just use data length * percent to get training dataset.
    Argument:
        x -- ndarray data 
        y -- ndarray label
        p -- percent of training dataset, should be 0~1
    Returns:
        ndarray -- x_train 4D ndarray
        ndarray -- x_valid 4D ndarray
        ndarray -- y_train 2D ndarray of shape (encode, train nums)
        ndarray -- y_valid 2D ndarray of shape (encode, valid nums)
    '''
    threshold = int(len(x) * p)
    return x[:threshold], x[threshold:], y[:threshold].T, y[threshold:].T