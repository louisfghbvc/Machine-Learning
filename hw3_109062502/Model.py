import numpy as np
import matplotlib.pyplot as plt
from Function import *

class Model:
    def __init__(self, layerList, batch_size = 60, max_epoch = 15, learning_rate = 0.1):
        '''
        Introduction:
            Class initial method.
        Argument:
            layerList -- layer of model.
            batch_size -- each train data size 
            max_epoch -- maximum train loop of iterate all training data 
            learning_rate -- gradient learning rate
        '''
        self.layerList = layerList
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate

    def forward(self, x):
        '''
        Introduction:
            Forward all layer in layerList.
            if i == 0: means first layer so use input x
            else: use previous layer activation output
        Argument:
            x -- 4D ndarray of input data e.g (batch_size, 32, 32, 3)
        Returns:
            ndarray -- 2D ndarray output of model e.g (nums of label, batch_size) 
        '''
        for i, layer in enumerate(self.layerList):
            if not i: layer.forward(x)
            else: layer.forward(self.layerList[i-1].a)
        return self.layerList[-1].a

    def backward(self, x, y):
        '''
        Introduction:
            Backward all layer in layerList. Reversed iterate all layer.
            Pass prevLayer, nextLayer information to each layer.
            if i == first: previous layer use input data.
            if i == last: next layer use ouput label.
            otherwise just pass prev, next = layer[i-1], layer[i+1]
        Argument:
            x -- 4D ndarray of input data e.g (batch_size, 32, 32, 3)
            y -- 2D ndarray of output data e.g (nums of label names, batch_size)
        '''
        for i in range(len(self.layerList)-1, -1, -1):
            prev_x = x if i == 0 else self.layerList[i-1]
            next_y = y if i == len(self.layerList)-1 else self.layerList[i+1]
            self.layerList[i].backward(prev_x, next_y)

    def updateGrad(self, batch_size):
        '''
        Introduction:
            Update all layer weight, bias.
        Argument:
            batch_size -- nums of dataset
        '''
        for layer in self.layerList:
            layer.w -= self.learning_rate * (1./batch_size) * layer.grad['dw']
            layer.b -= self.learning_rate * (1./batch_size) * layer.grad['db']

    def train(self, x, y, x_valid, y_valid):
        '''
        Introduction:
            Training model.
        Argument:
            x -- 4D ndarray training data of shape e.g. (nums of train, 32, 32, 3)
            y -- 2D ndarray training label. e.g. (label nums, nums of train)
            x_valid -- 4D ndarray validation data e.g. (nums of valid, 32, 32, 3)
            y_valid -- 2D ndarray validation label e.g. (label nums, nums of valid)
        '''
        
        # Use for plotting loss
        plot_x = [i for i in range(self.max_epoch)]
        train_lt = []
        valid_lt = []

        for epoch in range(self.max_epoch):
            # each epoch suffle dataset. transpose to make shape corrected
            rx, ry = SuffleXY(x, y.T)
            ry = ry.T

            # each mini-batch, make batch_size = min(batch_size, remain data number)
            for i in range(0, (x.shape[0] + self.batch_size - 1)//self.batch_size):
                # Get batch
                x_batch = rx[0+i*self.batch_size : self.batch_size+i*self.batch_size, :]
                y_batch = ry[:, 0+i*self.batch_size : self.batch_size+i*self.batch_size]

                self.forward(x_batch)
                self.backward(x_batch, y_batch)
                self.updateGrad(x_batch.shape[0])

            # train and valid output
            train_p = self.forward(x)
            valid_p = self.forward(x_valid)

            # Calculate loss, use CrossEncropy
            train_loss = CrossEncropy(y, train_p) / x.shape[0]
            valid_loss = CrossEncropy(y_valid, valid_p) / x_valid.shape[0]
            print('epoch: {}, train_loss: {:.6f}, validation_loss: {:.6f}, val_acc: {:.4f}'.format(epoch+1, train_loss, valid_loss, CalAcc(valid_p, y_valid)))

            # Record loss
            train_lt.append(train_loss)
            valid_lt.append(valid_loss)
        
        # plot loss graph
        plt.plot(plot_x, train_lt, lw=2, label='train_loss')
        plt.plot(plot_x, valid_lt, lw=2, label='valid_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title('Show loss')
        plt.show()
        plt.close()
        
    def predict(self, x, y):
        '''
        Introduction:
            Predict dataset. And calculate loss and accuracy
        Argument:
            x -- 4D ndarray test data of shape e.g. (nums of test, 32, 32, 3)
            y -- 2D ndarray test label. e.g. (label nums, nums of test)
        '''
        test_p = self.forward(x)
        test_loss = CrossEncropy(y, test_p) / x.shape[0]
        print('test_loss: {:.6f}, test_acc: {:.4f}'.format(test_loss, CalAcc(test_p, y)))
