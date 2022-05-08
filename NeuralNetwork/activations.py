import numpy as np


# Base Activation Class
class NonLinearFunction(object):
    '''
        NonLinearityFunction (Base Class)
            Base Class for Non-Linearity Function Classes.
    '''
    def __call__(self, z):
        raise NotImplementedError('Should be implemented by subclasses.')

    def backprop(self, prev_dy, *args, **kwargs):
        raise NotImplementedError('Should be implemented by subclasses.')
        

# Sigmoid
class Sigmoid(NonLinearFunction):
    '''
        Sigmoid Activation Function (Base Class: NonLinearFunction)
    '''
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', self.__class__.__name__)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __call__(self, z):
        self.z = z
        return self.__sigmoid(z)

    def backprop(self, prev_dy, *args, **kwargs):
        s = self.__sigmoid(self.z)
        d = s * (1 - s)
        return prev_dy * d


# Softmax
class Softmax(NonLinearFunction):
    '''
        Softmax Activation Function (Base Class: NonLinearFunction)
    '''
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', self.__class__.__name__)

    def __softmax(self, z):
        exps = np.exp(z - np.max(z, axis = -1, keepdims = True))
        return exps / np.sum(exps, axis = -1, keepdims = True)

    def __call__(self, z):
        self.z = z
        return self.__softmax(z)

    def backprop(self, prev_dy, *args, **kwargs):
        s = self.__softmax(self.z)
        s_len = len(s.shape)

        #d1 = np.expand_dims(s, axis = -1)
        #d2 = -(d1 @ np.transpose(d1, axes = [0, 1, 3, 2]))
        #d3 = d1 * np.eye(s.shape[-1])[np.newaxis, np.newaxis, :, :]
        #d4 = d3 + d2

        d1 = np.expand_dims(s, axis = -1)
        
        axes = list(np.arange(s_len - 1)) + list(np.arange(s_len, s_len - 2, -1))
        d2 = -(d1 @ np.transpose(d1, axes = axes))

        expd = np.eye(s.shape[-1])
        for _ in range(len(d1.shape) - len(expd.shape)):
            expd = np.expand_dims(expd, axis = 0)
        d3 = d1 * expd
        
        d4 = d3 + d2

        return np.squeeze(d4 @ np.expand_dims(prev_dy,axis = -1))


# Tanh
class Tanh(NonLinearFunction):
    '''
        Tanh Activation Function (Base Class: NonLinearFunction)
    '''
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', self.__class__.__name__)

    def __tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def __call__(self, z):
        self.z = z
        return self.__tanh(z)

    def backprop(self, prev_dy, *args, **kwargs):
        s = self.__tanh(self.z)
        d = (1 - s ** 2)
        return prev_dy * d



# ReLU
class ReLU(NonLinearFunction):
    '''
        ReLU (Rectified Linear Unit) Activation Function (Base Class: NonLinearFunction)
    '''
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', self.__class__.__name__)

    def __relu(self, z):
        return np.maximum(0, z)

    def __call__(self, z):
        self.z = z
        return self.__relu(z)

    def backprop(self, prev_dy, *args, **kwargs):
        d = np.where(self.z > 0, 1, 0)
        return prev_dy * d



# LeakyReLU
class LeakyReLU(NonLinearFunction):
    '''
        LeakyReLU (Base Class: NonLinearFunction)
    '''
    def __init__(self, alpha = 0.3, **kwargs):
        self.alpha = alpha
        self.name = kwargs.get('name', self.__class__.__name__)

    def __leaky_relu(self, z):
        return np.maximum(self.alpha, z)

    def __call__(self, z):
        self.z = z
        return self.__leaky_relu(z)

    def backprop(self, prev_dy, *args, **kwargs):
        d = np.where(self.z > self.alpha, 1, 0)
        return prev_dy * d
        


# Activation Class
class Activation(Sigmoid, Softmax, Tanh, ReLU, LeakyReLU):
    '''
        Activation (Base Class: Sigmoid, Softmax, Tanh, ReLU, LeakyReLU)
            Activation Layer through which all type of available activations
            can be implemented.

        Args:
            activation_name:
                    - type: string
                    - about: Defines acvtivation function.
    '''
    def __init__(self, activation_name):
        self.activation_name = activation_name

        if activation_name == 'sigmoid':
            super(Activation, self).__init__()
        elif activation_name == 'softmax':
            super(Sigmoid, self).__init__()
        elif activation_name == 'tanh':
            super(Softmax, self).__init__()
        elif activation_name == 'relu':
            super(Tanh, self).__init__()
        elif activation_name == 'leaky_relu':
            super(ReLU, self).__init__()

    @property
    def __activations(self):
        activations = {
                'sigmoid': super(Activation, self),
                'softmax': super(Sigmoid, self),
                'tanh': super(Softmax, self),
                'relu': super(Tanh, self),
                'leaky_relu': super(ReLU, self)
            }
        return activations
        
    def __call__(self, z):
        self.z = z
        self.activation = self.__activations[self.activation_name]
        return self.activation.__call__(z)

    def backprop(self, prev_dy, *args, **kwargs):
        return self.activation.backprop(prev_dy)



