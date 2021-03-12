import numpy as np

class Dense:
    def __init__(self, n_output, activative_func = None):
        '''
        Introduction:
            Class initial method.
        Argument:
            n_output -- output neuron numbers
            activation_func -- any function like method.
        '''
        self.w = None
        self.b = np.zeros((n_output, 1))
        self.n_output = n_output
        self.af = activative_func
        self.grad = None

    def forward(self, x):
        '''
        Introduction:
            Model forward.
        Argument:
            x -- 2D ndarray input data
        Returns:
            ndarray -- 2D ndarray output of model e.g (nums of output neurons, batch_size) 
        '''
        # Only do first time, set weight
        if self.w is None:
            n_input = x.shape[0]
            self.w = np.random.randn(self.n_output, n_input) * np.sqrt(1. / n_input)

        self.z = self.w.dot(x) + self.b
        self.a = self.af(self.z)
        return self.a
    
    def backward(self, prevL, nextL):
        '''
        Introduction:
            Model backward.
        Argument:
            prevL -- previous layer, maybe ndarray or layer
            nextL -- next layer, maybe ndarray or layer
        '''
        prev_x = prevL if isinstance(prevL, np.ndarray) else prevL.a
        if isinstance(nextL, np.ndarray):
            # CrossEncropy + Softmax, just fix.
            dz = self.a - nextL
        else:
            da = nextL.grad['dx']
            # Derivative of Activation
            dz = self.af.backward(da, self.z)

        # Chain rule, send to previous layer
        dx = self.w.T.dot(dz)
        # Derivative w, b
        dw = dz.dot(prev_x.T)
        db = np.sum(dz, axis=1, keepdims=True)

        self.grad = {'dx': dx, 'dw': dw, 'db': db}