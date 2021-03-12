import numpy as np
from Function import *
from Model import *
from Dense import *
from Conv import *
from Flatten import *

if __name__ == "__main__":
    # Train data read and suffle
    data, label = ReadData(isTrain=True, labelList = ['Carambula', 'Lychee', 'Pear'])
    data, label = SuffleXY(data, label)

    # Build model
    conv2d1 = Conv(kernel_size=3, in_channel=3, out_channel=2, padding='no', step=1, activation_func=None)
    flat = Flatten()
    dense1 = Dense(n_output=100, activative_func=ReLU())
    dense2 = Dense(n_output=3, activative_func=Softmax)

    model = Model(layerList=[conv2d1, flat, dense1, dense2], learning_rate=0.01, batch_size=60, max_epoch=60)

    # Split dataset into train and valid, p is the percent of train data
    x_train, x_valid, y_train, y_valid = SplitData(data, label, p = 0.7)

    # Start Training
    model.train(x_train, y_train, x_valid, y_valid)

    # Testing data, and predict
    test_data, test_label = ReadData(isTrain=False, labelList = ['Carambula', 'Lychee', 'Pear'])
    model.predict(test_data, test_label.T)
