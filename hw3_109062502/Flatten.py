import numpy as np

class Flatten:
    def __init__(self):
        '''
        Introduction:
            Flatten initial method. usefor conv and dense layer connection
            gradient just dummy none.
        '''
        self.grad = None
        self.w = 0
        self.b = 0

    def forward(self, x):
        '''
        Introduction:
            Transpose x to (..., batch_size). So that dense layer can accept
        Argument:
            x -- 4D ndarray of shape e.g. (batch_size, height, width, channel)
        Returns:
            ndarray -- activation ouput of x
        '''
        self.x_shape = x.shape
        nums, h, w, c = x.shape
        self.a = x.T.reshape((h*w*c, nums))
        return self.a

    def backward(self, prevL, nextL):
        '''
        Introduction:
            Backward. make nextL gradient to origin conv shape
            Only use dx.
        Argument:
            prevL -- previous layer information, no use.
            nextL -- next layer information
        '''
        dx = nextL.grad['dx'].T.reshape(self.x_shape)
        self.grad = {'dx': dx, 'dw': 0, 'db' : 0}