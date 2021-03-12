import numpy as np

class Conv:
    def __init__(self, kernel_size, in_channel, out_channel, step = 1, padding = 'no', activation_func = None):
        '''
        Introduction:
            Class initial method.
        Argument:
            kernel_size -- filter map size. (h,w) is same
            in_channel -- input data channel
            out_channel -- output data channel
            step -- window step, default = 1
            padding -- 'same' is after conv, input (w,h) = output (w,h). default is 'no' 
            activation_func -- any function like method.
        '''
        self.kernel_size = kernel_size
        self.channel = out_channel
        self.w = np.random.randn(kernel_size, kernel_size, in_channel, out_channel) * 0.1
        self.b = np.zeros((out_channel, 1))
        self.af = activation_func
        self.step = step
        self.padding = padding
        self.grad = None

    def padData(self, x, pad):
        '''
        Introduction:
            Padding data x according to pad.
            Use np.pad.
        Argument:
            x -- 4D ndarray data 
            pad -- tuple (h, w)
        Returns:
            ndarray -- after padding 4D ndarray data
        '''
        return np.pad(x, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), 'constant')
    
    def genWindow(self, x, hw):
        '''
        Introduction:
            Generate sliding window need to convolution
        Argument:
            x -- 4D ndarray data 
        Returns:
            tuple (ndarray, i, j) -- 4D ndarray window size. and index of window.
        '''
        for i in range(hw[0]):
            for j in range(hw[1]):
                h_start = i * self.step
                h_end = h_start + self.kernel_size
                w_start = j * self.step
                w_end = w_start + self.kernel_size
                yield x[:, h_start:h_end, w_start:w_end, :], i, j

    def lazyCalPadShape(self, x):
        '''
        Introduction:
            Determine how to padding. and get padding shape 
        Argument:
            x -- 4D ndarray data 
        Returns:
            tuple (h, w) -- how to padding according to h, w 
        '''
        if self.padding == 'no':
            need_pad = (0, 0)
        elif self.padding == 'same':
            need_pad = self.calPadShape(x.shape, x.shape)
        else:
            raise ValueError("padding not valid !!")
        return need_pad

    def calPadShape(self, in_shape, out_shape):
        '''
        Introduction:
            Calculate padding shape. according to input and output shape
            formula is simple to calculate. 
            Say x is what we need.
            (x - kernel_size) / step + 1 = output.
            and calculate from that.
        Argument:
            x -- 4D ndarray data 
        Returns:
            tuple (h, w) -- how to padding according to h, w 
        '''
        _, ih, iw, _ = in_shape
        _, oh, ow, _ = out_shape
        fat_h = (oh - 1) * self.step + self.kernel_size
        fat_w = (ow - 1) * self.step + self.kernel_size
        return (fat_h - ih) // 2, (fat_w - iw) // 2

    def forward(self, x):
        '''
        Introduction:
            Model Convolution forward.
        Argument:
            x -- 4D ndarray data 
        Returns:
            ndarray -- 4D ndarray data
        '''
        x = self.padData(x, self.lazyCalPadShape(x))
        nums, h, w, _ = x.shape
        self.a = np.zeros((nums, (h-self.kernel_size) // self.step + 1, (w-self.kernel_size) // self.step + 1, self.channel))     

        for sub_mat, i, j in self.genWindow(x, (self.a.shape[1], self.a.shape[2])):
            for c in range(self.channel):
                self.a[:, i, j, c] = np.sum(sub_mat * self.w[np.newaxis, :, :, :, c], axis=(1, 2, 3)) + self.b[c]

        if self.af: 
            self.z = self.a
            self.a = self.af(self.z)

        return self.a

    def backward(self, prevL, nextL):
        '''
        Introduction:
            Model Convolution backward.
            First, calculate activation function backward.
            Second, calculate dw, db. use window * next layer gradient, same as convolution .
            Third, calculate dx. need to rotate weight 180 degree. swap input and output channel,
            and padding on next layer gradient.
            then convolution with rotated weight. finally get dx
        Argument:
            x -- 4D ndarray data 
        Returns:
            ndarray -- 4D ndarray data
        '''
        prev_x = prevL if isinstance(prevL, np.ndarray) else prevL.a
        prev_x = self.padData(prev_x, self.lazyCalPadShape(prev_x))

        # From next layer gradient
        next_dx = nextL.grad['dx']

        # Calculate activation backward
        if self.af: next_dx = self.af.backward(next_dx, self.z)

        dx = np.zeros_like(prev_x)
        dw = np.zeros_like(self.w)
        db = np.zeros_like(self.b)

        # Calculate dw, db
        for sub_mat, i, j in self.genWindow(prev_x, (self.a.shape[1], self.a.shape[2])):
            for c in range(self.channel):
                dw[:, :, :, c] += np.sum(sub_mat * next_dx[:, i, j, c, np.newaxis, np.newaxis, np.newaxis], axis=0)
                db[c] += np.sum(next_dx[:, i, j, c])
        
        rot_w = self.w[::-1, ::-1, :, :].swapaxes(2, 3)
        need_pad = self.calPadShape(in_shape=next_dx.shape, out_shape=prev_x.shape)
        next_dx = self.padData(next_dx, need_pad) 

        # Calculate dx, to prevous layer
        for sub_mat, i, j in self.genWindow(next_dx, (prev_x.shape[1], prev_x.shape[2])):
            for c in range(prev_x.shape[3]):
                dx[:, i, j, c] = np.sum(sub_mat * rot_w[np.newaxis, :, :, :, c], axis=(1, 2, 3))

        self.grad = {'dx': dx, 'dw': dw, 'db' : db}