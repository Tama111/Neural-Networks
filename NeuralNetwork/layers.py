import numpy as np
from NeuralNetwork.activations import Activation
from copy import copy
from NeuralNetwork.initializer import WeightsInitializer

#######################################################################


# ==============================================================================================================================
#    Base (Classes) Layers
# ==============================================================================================================================


# Base class Layer
class Layer(object):
    '''
        Base class Layer
        All the layers are made as it being its base class.
    '''
    def preset_optimizer(self, optimizer):
        raise NotImplementedError()

    def build(self, input_shape):
        raise NotImplementedError()

    def __call__(self, inputs):
        raise NotImplementedError()

    def backprop(self, prev_dy):
        raise NotImplementedError()

    @property
    def parameters(self):
        if len(self.all_weights):
            params = []
            for w in self.all_weights:
                params.append(np.prod(w.shape))
                
            return np.int(np.sum(params))
        return 0

    @property
    def trainable_parameters(self):
        if len(self.all_weights):
            if self.trainable:
                params = []
                for w in self.all_weights:
                    if w.trainable:
                        params.append(np.prod(w.shape))
                        
                return np.int(np.sum(params))
            return 0
        return 0
        

    @property
    def trainable_weights(self):
        weights_trainable = []
        if len(self.all_weights):
            if self.trainable:
                for w in self.all_weights:
                    if w.trainable:
                        weights_trainable.append(w.weight)

                return weights_trainable
            return weights_trainable
        return weights_trainable

    @property
    def output_shape(self):
        raise NotImplementedError()

    def _update_weights(self, opt_wgt_grad):
        for opt, weight, grad in opt_wgt_grad:
            if weight.trainable:
                opt(weight.weight, grad)


#####################################################################################################

# Base Class Pad2D
class Pad2D(object):
    '''
        Padding 2D Layer (Base Class)
            It is a base class for all those layers having padding feature/functionality.
    '''
    def _padding(self, kernel_size = None, strides = None, padding = None):

        if isinstance(padding, str):
            pad = np.array(kernel_size) - np.array(strides)

            if padding == 'valid':
                return (0, 0), (0, 0)

            elif padding == 'same':
                h0 = np.int(np.floor(pad[0]/2))
                h1 = np.int(np.ceil(pad[0]/2))
                w0 = np.int(np.floor(pad[1]/2))
                w1 = np.int(np.ceil(pad[1]/2))
                pad_h = (h0, h1)
                pad_w = (w0, w1)

            #assert (pad[0] >= 0) & (pad[1] >= 0)

            if pad[0] <= 0:
                pad_h0 = np.int(np.floor((kernel_size[0] - 1)/2))
                pad_h1 = np.int(np.ceil((kernel_size[0] - 1)/2))
                pad_h = (pad_h0, pad_h1)
                
            if pad[1] <= 0:
                pad_w0 = np.int(np.floor((kernel_size[1] - 1)/2))
                pad_w1 = np.int(np.ceil((kernel_size[1] - 1)/2))
                pad_w = (pad_w0, pad_w1)

            return pad_h, pad_w

        elif isinstance(padding, int):
            pad_h = (padding, padding)
            pad_w = (padding, padding)

            return pad_h, pad_w

        elif isinstance(padding, tuple):
            if isinstance(padding[0], int):
                pad_h = (padding[0], padding[0])
                pad_w = (padding[1], padding[1])
                return pad_h, pad_w

            elif isinstance(padding[0], tuple):
                return padding




# ==============================================================================================================================
#    Some Basic Layers
# ==============================================================================================================================
            

# Input Layer
class Input(Layer):
    '''
        Input Layer (Base Class: Layer)
            Input shape of the network is defined in this layer

        Args:
            input_shape:
                - type: tuple
                - about: A tuple with positive integers defining the input shape of the
                         network (or the first layer). Batch size is not defined in the input shape.
                         It is either denoted as None or not even denoted.

            **kwargs:
                    name:
                        - type: string
                        - about: Name of the layer can be defined.
                        - default: `Input`
        
    '''
    def __init__(self, input_shape: tuple, **kwargs):
        if input_shape[0] is not None:
            self.input_shape = (None, ) + input_shape
        else:
            self.input_shape = input_shape
        self.name = kwargs.get('name', self.__class__.__name__)


#####################################################################################################


# Reshape Layer
class Reshape(Layer):
    '''
        Reshape Layer (Base Class: Layer)
            This layer reshapes the input into a certain shape
            that is defined in the `shape` parameter.


        Args:
            shape:
                - type: tuple
                - about: define the shape in which the input should
                         be converted into.

            **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `Reshape`

        Input Shape:
            N-D tensor of shape (batch_size, ...)

        Output Shape:
            N-D tensor of shape (batch_size, ) + (shape, )

            
    '''
    def __init__(self, shape: tuple, **kwargs):

        if isinstance(shape, int):
            self.shape = (shape, )
        elif isinstance(shape, tuple):
            self.shape = shape
        
        self.input_shape = None
        self.all_weights = ()

        self.name = kwargs.get('name', self.__class__.__name__)

    def __call__(self, inputs):
        self.inputs = inputs
        if self.input_shape is None:
            self.input_shape = inputs.shape

        # reshaping the input as defined shape in forward propagation
        out = inputs.reshape((inputs.shape[0], ) + self.shape)

        assert out.shape[1:] == self.output_shape[1:]
        return out

    def backprop(self, prev_dy):
        # reshaping the gradient as input shape in backward propagation
        out = prev_dy.reshape(self.inputs.shape)
        assert out.shape == self.inputs.shape
        return out

    @property
    def output_shape(self):
        batch, inp_h, inp_w, inp_chn = self.input_shape
        return (batch, ) + self.shape



#####################################################################################################


# BatchNormalization Layer
class BatchNormalization(Layer):
    '''
        BatchNormalization Layer (Base Layer: Layer)
            This layer normalizes the input across batch.


        Args:
            loc:
                - type: boolean
                - about: Decide whether to apply/center the normalization.
                - default: True
                
            scale:
                - type: boolean
                - about: Decide whether to apply scale to normalization.
                - default: True
                
            **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `BatchNormalization`

                trainable:
                    - type: boolean
                    - about: Defines if the weights of the layer is trainable.
                    - default: True

                weight_initializer:
                    - type: string
                    - about: Appropriate Weight Initialization technique is defined.
                    - default: 'uniform'

        Input Shape:
            N-D tensor of shape (batch_size, ..., inp_dim)

        Output Shape:
            4-D tensor of shape (batch_size, ..., inp_dim)
            
    '''
    def __init__(self, loc: bool = True, scale: bool = True, **kwargs):
        #self.momentum = momentum
        self.loc = loc
        self.scale = scale
        self.epsilon = 1e-08
        
        self.input_shape = None
        self.all_weights = ()
        
        self.name = kwargs.get('name', self.__class__.__name__)
        self.trainable = kwargs.get('trainable', True)
        self.weight_initializer = kwargs.get('weight_initializer', 'uniform')

    def preset_optimizer(self, optimizer):
        self.gamma_opt = copy(optimizer)
        self.beta_opt = copy(optimizer)

    def build(self, input_shape):
        self.input_shape = input_shape
        _, inp_h, inp_w, inp_chn = input_shape
        
        self.gamma = WeightsInitializer(shape = (1, inp_h, inp_w, inp_chn), trainable = True,
                                        initializer = self.weight_initializer,
                                        name = f'{self.name}|Gamma:gamma')
        self.beta = WeightsInitializer(shape = (1, inp_h, inp_w, inp_chn),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|Beta:beta')

        self.all_weights = (self.gamma, self.beta)

    def __call__(self, inputs):
        self.inputs = inputs
        if not hasattr(self, 'beta'):
            self.build(inputs.shape)

        '''
        if self.moving_mean is None:
            self.moving_mean = np.expand_dims(np.mean(self.inputs, axis = 0), axis = 0)
        if self.moving_std is None:
            self.moving_std = np.expand_dims(np.std(self.inputs, axis = 0), axis = 0)

        if self.trainable & training:
            mean = np.expand_dims(np.mean(self.inputs, axis = 0), axis = 0)
            var = np.expand_dims(np.std(self.inputs, axis = 0), axis = 0)
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
            self.moving_std =  self.momentum * self.moving_std + (1 - self.momentum) * std

        else:
            mean = self.moving_mean
            std = self.moving_std
        '''

        # Forward Propagation
        self.mean = np.mean(self.inputs, axis = 0, keepdims = True)
        self.std = np.std(self.inputs, axis = 0, keepdims = True)
        self.x_norm = (self.inputs - self.mean) / (self.std + self.epsilon)

        if (self.scale == True) & (self.loc == True):
            out = self.gamma.weight * self.x_norm + self.beta.weight
        elif (self.scale == True) & (self.loc == False):
            out = self.gamma.weight * self.x_norm
        elif (self.scale == False) & (self.loc == True):
            out = self.x_norm + self.beta.weight
        else:
            out = self.x_norm

        assert out.shape[1:] == self.output_shape[1:]
        return out
    
    def backprop(self, prev_dy):

        # BackPropagation
        if self.trainable:
            d_gamma, d_beta = np.zeros_like(self.gamma.weight), np.zeros_like(self.beta.weight)
            if self.scale:
                d_gamma += np.sum(prev_dy * self.x_norm, axis = 0, keepdims = True)
            if self.loc:
                d_beta += np.sum(prev_dy, axis = 0, keepdims = True)

            if (self.scale == True) & (self.loc == True):
                grad_pairs = [(self.gamma_opt, self.gamma, d_gamma),
                          (self.beta_opt, self.beta, d_beta)]
            elif (self.scale == True) & (self.loc == False):
                grad_pairs = [(self.gamma_opt, self.gamma, d_gamma)]
            elif (self.scale == False) & (self.loc == True):
                grad_pairs = [(self.beta_opt, self.beta, d_beta)]
            else:
                grad_pairs = []

            if len(grad_pairs) > 0:
                self._update_weights(grad_pairs)


        if (self.scale == True) & (self.loc == True):
            dx = prev_dy * self.gamma.weight / (self.std + self.epsilon)
        elif (self.scale == True) & (self.loc == False):
            dx = prev_dy * self.gamma.weight / (self.std + self.epsilon)
        elif (self.scale == False) & (self.loc == True):
            dx = prev_dy / (self.std + self.epsilon)
        else:
            dx = prev_dy / (self.std + self.epsilon)

        assert dx.shape == self.inputs.shape
        return dx


    @property
    def output_shape(self):
        return self.input_shape


#####################################################################################################


# InstanceNormalization Layer
class InstanceNormalization(Layer):
    '''
        InstanceNormalization Layer (Base Layer: Layer)
            This layer normalizes the input across height and width.


        Args:
            loc:
                - type: boolean
                - about: Decide whether to apply/center the normalization.
                - default: True
                
            scale:
                - type: boolean
                - about: Decide whether to apply scale to normalization.
                - default: True
                
            **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `InstanceNormalization`

                trainable:
                    - type: boolean
                    - about: Defines if the weights of the layer is trainable.
                    - default: True

                weight_initializer:
                    - type: string
                    - about: Appropriate Weight Initialization technique is defined.
                    - default: 'uniform'

        Input Shape:
            N-D tensor of shape (batch_size, ..., inp_dim)

        Output Shape:
            4-D tensor of shape (batch_size, ..., inp_dim)
            
    '''
    def __init__(self, loc: bool = True, scale: bool = True, **kwargs):
        self.loc = loc
        self.scale = scale
        self.epsilon = 1e-08
        
        self.input_shape = None
        self.all_weights = ()
        
        self.name = kwargs.get('name', self.__class__.__name__)
        self.trainable = kwargs.get('trainable', True)
        self.weight_initializer = kwargs.get('weight_initializer', 'uniform')

    def preset_optimizer(self, optimizer):
        self.gamma_opt = copy(optimizer)
        self.beta_opt = copy(optimizer)

    def build(self, input_shape):
        self.input_shape = input_shape
        inp_chn = input_shape[-1]
        
        self.gamma = WeightsInitializer(shape = (1, 1, 1, inp_chn), trainable = True,
                                        initializer = self.weight_initializer,
                                        name = f'{self.name}|Gamma:gamma')
        self.beta = WeightsInitializer(shape = (1, 1, 1, inp_chn),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|Beta:beta')

        self.all_weights = (self.gamma, self.beta)

    def __call__(self, inputs):
        self.inputs = inputs
        if not hasattr(self, 'beta'):
            self.build(inputs.shape)

        # Forward Propagation
        self.mean = np.mean(self.inputs, axis = (1, 2), keepdims = True)
        self.std = np.std(self.inputs, axis = (1, 2), keepdims = True)
        self.x_norm = (self.inputs - self.mean) / (self.std + self.epsilon)

        if (self.scale == True) & (self.loc == True):
            out = self.gamma.weight * self.x_norm + self.beta.weight
        elif (self.scale == True) & (self.loc == False):
            out = self.gamma.weight * self.x_norm
        elif (self.scale == False) & (self.loc == True):
            out = self.x_norm + self.beta.weight
        else:
            out = self.x_norm

        assert out.shape[1:] == self.output_shape[1:]
        return out
    
    def backprop(self, prev_dy):

        # BackPropagation
        if self.trainable:
            d_gamma, d_beta = np.zeros_like(self.gamma.weight), np.zeros_like(self.beta.weight)
            print(d_gamma.shape, d_beta.shape)
            if self.scale:
                d_gamma += np.sum(prev_dy * self.x_norm, axis = (0, 1, 2), keepdims = True)
            if self.loc:
                d_beta += np.sum(prev_dy, axis = (0, 1, 2), keepdims = True)

            if (self.scale == True) & (self.loc == True):
                grad_pairs = [(self.gamma_opt, self.gamma, d_gamma),
                          (self.beta_opt, self.beta, d_beta)]
            elif (self.scale == True) & (self.loc == False):
                grad_pairs = [(self.gamma_opt, self.gamma, d_gamma)]
            elif (self.scale == False) & (self.loc == True):
                grad_pairs = [(self.beta_opt, self.beta, d_beta)]
            else:
                grad_pairs = []

            if len(grad_pairs) > 0:
                self._update_weights(grad_pairs)


        if (self.scale == True) & (self.loc == True):
            dx = prev_dy * self.gamma.weight / (self.std + self.epsilon)
        elif (self.scale == True) & (self.loc == False):
            dx = prev_dy * self.gamma.weight / (self.std + self.epsilon)
        elif (self.scale == False) & (self.loc == True):
            dx = prev_dy / (self.std + self.epsilon)
        else:
            dx = prev_dy / (self.std + self.epsilon)

        assert dx.shape == self.inputs.shape
        return dx


    @property
    def output_shape(self):
        return self.input_shape


######################################################################################################


# Dropout Layer
class Dropout(Layer):
    '''
        Dropout Layer (Base Class: Layer)
            It is a simple dropout layer.


        Args:
            rate:
                - type: float
                - about: defines the amount of dropout that needed to be applied.
                         Its value should range between 0 and 1.

            **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `Dropout`

        Input Shape:
            4-D tensor with shape (batch_size, inp_height, inp_width, inp_feature_maps)

        Output Shape:
            4-D tensor with shape (batch_size, inp_height, inp_width, inp_feature_maps)
                
    '''
    def __init__(self, rate: int, **kwargs):
        self.rate = rate
        
        self.input_shape = None
        self.all_weights = ()

        self.name = kwargs.get('name', self.__class__.__name__)

    def __call__(self, inputs):
        self.inputs = inputs
        if self.input_shape is None:
            self.input_shape = inputs.shape

        # forward propagation
        self.dropout_mask = np.random.choice([0, 1], size = self.inputs.shape,
                                        p = [self.rate, 1 - self.rate])

        # self.dropout_mask = np.where(np.random.uniform(low = 0, high = 1,
        #                              size = self.input_shape) < self.rate, 0, 1)

        out = self.inputs * self.dropout_mask

        assert out.shape[1:] == self.output_shape[1:]
        return out


    def backprop(self, prev_dy):

        # backpropagation
        out = prev_dy * self.dropout_mask
        assert out.shape == self.inputs.shape
        return out
    

    @property
    def output_shape(self):
        return self.input_shape
        
        

# ==============================================================================================================================
#    ANN (Artificial Neural Network)
# ==============================================================================================================================

# Linear Layer
class Linear(Layer):
    '''
        Linear Layer (Base Class: Layer)
            A simple Linear layer, helps in making fully-connected layers or ANNs.

        Args:
            neurons:
                - type: +ve integer
                - about: A positive integer, defining output dimensionality.
                         Number of units/neurons in the layer is defined.

            activation:
                - type: string
                - about: A string value defining the appropriate activation function name.
                - default: None

            **kwargs:
                    name:
                        - type: string
                        - about: Name of the layer can be defined.
                        - default: `Linear`

                    trainable:
                        - type: boolean
                        - about: Defines if the weights of the layer is trainable.
                        - default: True

                    weight_initializer:
                        - type: string
                        - about: Appropriate Weight Initialization technique is defined.
                        - default: 'uniform'


        Input Shape:
            N-D tensor with shape (batch_size, ..., input_dims)

        Output Shape:
            N-D tensor with shape (batch_size, ..., neurons)
        
    '''
    def __init__(self, neurons: int, activation: str = None, **kwargs):
        self.neurons = neurons
        self.activation = activation
        
        self.input_shape = None
        self.all_weights = ()

        self.name = kwargs.get('name', self.__class__.__name__)
        self.trainable = kwargs.get('trainable', True)
        self.weight_initializer = kwargs.get('weight_initializer', 'uniform')

    
    def preset_optimizer(self, optimizer):
        self.Wx_opt = copy(optimizer)
        self.Wb_opt = copy(optimizer)
        
        
    def build(self, input_shape):
        self.input_shape = input_shape
        self.Wx = WeightsInitializer(shape = (input_shape[-1], self.neurons),
                                    initializer = self.weight_initializer, trainable = True,
                                     name = f'{self.name}|Weight:Wx')
        self.Wb = WeightsInitializer(shape = (1, self.neurons),
                                     initializer = 'zeros', trainable = True,
                                     name = f'{self.name}|Bias:Wb')

        self.all_weights = (self.Wx, self.Wb)
        
                
    def __call__(self, inputs):
        self.inputs = inputs
        if not hasattr(self, 'Wb'):
            self.build(inputs.shape)

        z = inputs @ self.Wx.weight + self.Wb.weight
        self.act = Activation(self.activation)
        out = self.act(z) if self.activation is not None else z
        
        assert out.shape[1:] == self.output_shape[1:]
        return out
    
    def backprop(self, prev_dy):
        dp = self.act.backprop(prev_dy) if self.activation is not None else prev_dy

        if self.trainable:
            axes = list(np.arange(len(self.inputs.shape))[:-2]) + [-1, -2]
            dWx = np.transpose(self.inputs, axes = axes) @ dp
            dWb = np.sum(dp, axis = tuple(np.arange(len(dp.shape)))[:-1])

            if len(dWx.shape) == 3:
                dWx = np.sum(dWx, axis = 0)
            
            grad_pairs = [(self.Wx_opt, self.Wx, dWx), (self.Wb_opt, self.Wb, dWb)]
            self._update_weights(grad_pairs)

        dx = dp @ self.Wx.weight.T
        
        assert dx.shape == self.inputs.shape
        return dx

    @property
    def output_shape(self):
        return self.input_shape[:-1] + (self.neurons, )




# ==============================================================================================================================
#    RNN (Recurrent Neural Network)
# ==============================================================================================================================


# Simple RNN (Recurrent Neural Network) Layer
class RNN(Layer):
    '''
        RNN(Recurrent Neural Network) Layer (Base Class: Layer)
            A simple RNN, mostly used for sequential model. There are also
            different types of RNN.


        Args:
            hidden_units:
                - type: +ve integer
                - about: Defines output dimensionality. The number of hidden
                         neurons in the layer.

            return_sequences:
                - type: boolean
                - about: Decides whether to return hidden state from every time step.
                         If set as `False`, then would return the hidden state of the
                         last time step, otherwise return the hidden state from each
                         time step.
                - default: False

            **kwargs:
                    name:
                        - type: string
                        - about: Name of the layer can be defined.
                        - default: `RNN`

                    trainable:
                        - type: boolean
                        - about: Defines if the weights of the layer is trainable.
                        - default: True

                    weight_initializer:
                        - type: string
                        - about: Appropriate Weight Initialization technique is defined.
                        - default: 'uniform'


        Input Shape:
            3-D tensor with shape (batch_size, sequence, input_dim)

        Output Shape:
            If return_sequences is set to True then,
                3-D tensor with shape (batch_size, sequence, hidden_units)
            otherwise,
                2-D tensor with shape (batch_size, hidden_units)
        
    '''
    def __init__(self, hidden_units: int, return_sequences: bool = False, **kwargs):
        self.hidden_units = hidden_units
        self.return_sequences = return_sequences
        
        self.input_shape = None
        self.all_weights = ()

        self.name = kwargs.get('name', self.__class__.__name__)
        self.trainable = kwargs.get('trainable', True)
        self.weight_initializer = kwargs.get('weight_initializer', 'uniform')

    def preset_optimizer(self, optimizer):
        self.Wx_opt = copy(optimizer)
        self.Wh_opt = copy(optimizer)
        self.Wb_opt = copy(optimizer)

    def build(self, input_shape):
        self.input_shape = input_shape
        _, _, inp_dim = input_shape
        self.Wx = WeightsInitializer(shape = (inp_dim, self.hidden_units),
                                     initializer = self.weight_initializer, trainable = True,
                                     name = f'{self.name}|Weight:Wx')

        self.Wh = WeightsInitializer(shape = (self.hidden_units, self.hidden_units),
                                     initializer = self.weight_initializer, trainable = True,
                                     name = f'{self.name}|RecurrentWeight:Wh')

        self.Wb = WeightsInitializer(shape = (1, self.hidden_units),
                                     initializer = 'zeros', trainable = True,
                                     name = f'{self.name}|Bias:Wb')

        self.all_weights = (self.Wx, self.Wh, self.Wb)


    def __call__(self, inputs):
        self.inputs = inputs
        
        if not hasattr(self, 'Wb'):
            self.build(inputs.shape)

        # initializing the first (or input) hidden state as zeros
        h = np.zeros((inputs.shape[0], self.hidden_units))
        self.prev_h = {0: h} # keeping track of hidden states (will be used in backprop)

        # forward propagation (loop over length of sequence)
        for i in range(inputs.shape[1]):
            h = np.tanh(inputs[:, i, :] @ self.Wx.weight + h @ self.Wh.weight + self.Wb.weight)
            self.prev_h[i + 1] = h

        # whether to return hidden state from each time step
        if self.return_sequences:
            out = np.transpose(np.array(list(self.prev_h.values())[1:]), axes = [1, 0, 2])
            assert out.shape[1:] == self.output_shape[1:]
            return out

        # Return hidden state from last time step
        assert h.shape[-1] == self.output_shape[-1]
        return h

    def backprop(self, prev_dy):

        # initializing derivative of weights with zeros
        if self.trainable:
            dWx, dWh = np.zeros(self.Wx.shape), np.zeros(self.Wh.shape)
            dWb = np.zeros(self.Wb.shape)


        # Backpropagation
        if self.return_sequences:
            dh_ahead = np.zeros((self.inputs.shape[0], self.hidden_units))
        else:
            dh = prev_dy

        dx = []
        for t in reversed(range(self.input_shape[1])):
            if self.return_sequences:
                dh = (prev_dy[:, t, :] + dh_ahead) * (1 - self.prev_h[t + 1] ** 2)
            else:
                dh *= (1 - self.prev_h[t + 1] ** 2)

            if self.trainable:
                dWx += self.inputs[:, t, :].T @ dh
                dWh += self.prev_h[t].T @ dh
                dWb += np.sum(dh, axis = 0)

            if self.return_sequences:
                dh_ahead = dh @ self.Wh.weight.T
            else:
                dh = dh @ self.Wh.weight.T

            dx.append(dh @ self.Wx.weight.T)

        dx = np.transpose(np.array(dx[::-1]), axes = [1, 0, 2])

        # Updating the weights using optimizers
        if self.trainable:
            grad_pairs = [(self.Wx_opt, self.Wx, dWx), (self.Wh_opt, self.Wh, dWh),
                          (self.Wb_opt, self.Wb, dWb)]
            self._update_weights(grad_pairs)

        assert dx.shape == self.inputs.shape
        return dx

    @property
    def output_shape(self):
        batch, inp_seq, inp_dim = self.input_shape
        if self.return_sequences:
            return (batch, self.input_shape[1], self.hidden_units)
        return (batch, self.hidden_units)

    
#####################################################################################################


# BidirectionalRNN Layer
# Note: BidiretionalRNN Layer could have also been implemented
# by calling two RNN Layers in the below Layer, which could
# have reduced the work load.
class BidirectionalRNN(Layer):
    '''
        BidirectionalRNN Layer (Base Class: Layer)
            It is a Bidirectional Simple RNN Layer. Where inputs move from both
            directions, left-right & right-left, so as to get the information
            from the whole sentence.


        Args:
            hidden_units:
                - type: +ve integer / tuple of +ve integers
                - about: Defines output dimensionality. The number of hidden
                         neurons in the layer. If the input is tuple of shape 2,
                         then first integer defines the forward hidden units whereas,
                         the other defines the backward hidden units. If the input is
                         just an integer then, both forward and backward hidden units
                         have same number of neurons.

        **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `BidirectionalRNN`

                trainable:
                    - type: boolean
                    - about: Defines if the weights of the layer is trainable.
                    - default: True

                weight_initializer:
                    - type: string
                    - about: Appropriate Weight Initialization technique is defined.
                    - default: 'uniform'

        Input Shape:
            3-D tensor of shape (batch_size, sequence, input_dim)

        Output Shape:
            If hidden units is integer, then
                3-D tensor of shape (batch_size, sequence, 2 * hidden_units)
            otherwise, if hidden units is tuple, then,
                3-D tensor of shape (batch_size, sequence, hidden_units[0] + hidden_units[1])

        
    '''
    def __init__(self, hidden_units, **kwargs):
        self.input_shape = None
        self.all_weights = ()
        
        if type(hidden_units) == int:
            self.hidden_units_f = hidden_units
            self.hidden_units_b = hidden_units
        elif (type(hidden_units) == list) | (type(hidden_units) == tuple):
            self.hidden_units_f = hidden_units[0]
            self.hidden_units_b = hidden_units[1]


        self.name = kwargs.get('name', self.__class__.__name__)
        self.trainable = kwargs.get('trainable', True)
        self.weight_initializer = kwargs.get('weight_initializer', 'uniform')


    def preset_optimizer(self, optimizer):
        self.FWx_opt, self.FWh_opt, self.FWb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.BWx_opt, self.BWh_opt, self.BWb_opt = copy(optimizer), copy(optimizer), copy(optimizer)


    def build(self, input_shape):
        self.input_shape = input_shape
        _, _, inp_dim = input_shape

        # Forward Weights
        self.FWx = WeightsInitializer(shape = (inp_dim, self.hidden_units_f),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|ForwardWeight:FWx')
        self.FWh = WeightsInitializer(shape = (self.hidden_units_f, self.hidden_units_f),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|ForwardRecurrentWeight:FWh')
        self.FWb = WeightsInitializer(shape = (1, self.hidden_units_f),
                                      initializer = 'zeros', trainable = True,
                                      name = f'{self.name}|ForwardBias:FWb')


        # Backward Weights
        self.BWx = WeightsInitializer(shape = (inp_dim, self.hidden_units_b),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|BackwardWeight:BWx')
        self.BWh = WeightsInitializer(shape = (self.hidden_units_b, self.hidden_units_b),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|BackwardRecurrentWeight:BWh')
        self.BWb = WeightsInitializer(shape = (1, self.hidden_units_b),
                                      initializer = 'zeros', trainable = True,
                                      name = f'{self.name}|BackwardBias:BWb')

        self.all_weights = (self.FWx, self.FWh, self.FWb, self.BWx, self.BWh, self.BWb)


    def __tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


    def __call__(self, inputs):
        self.inputs = inputs
        if not hasattr(self, 'BWb'):
            self.build(inputs.shape)


        # initializing hidden states with zeros,
        # for both forward & backward direction
        fh = np.zeros((inputs.shape[0], self.hidden_units_f))
        bh = np.zeros((inputs.shape[0], self.hidden_units_b))

        # storing hidden states in each time step (forward & backward)
        self.prev_fh = {0: fh}
        self.prev_bh = {0: bh}

        inputs_rev = inputs[:, ::-1, :]

        # Forward propagation
        for t in range(self.input_shape[1]):

            # in forward direction
            fh = self.__tanh(inputs[:, t, :] @ self.FWx.weight + fh @ self.FWh.weight + self.FWb.weight)
            self.prev_fh[t + 1] = fh

            # in backward direction
            bh = self.__tanh(inputs_rev[:, t, :] @ self.BWx.weight + bh @ self.BWh.weight + self.BWb.weight)
            self.prev_bh[t + 1] = bh


        # concatenating hidden states from both
        # forward and backward direction for get the
        # hidden state from each time step
        concat_h = []
        for f_, b_ in zip(list(self.prev_fh.values())[1:], list(self.prev_bh.values())[1:][::-1]):
            concat_h.append(np.concatenate([f_, b_], axis = 1))
            #assert concat_h[-1].shape == (self.inputs.shape[0], self.hidden_units_f + self.hidden_units_b)
        out = np.transpose(np.array(concat_h), axes = [1, 0, 2])

        assert out.shape[1:] == self.output_shape[1:]
        return out

    def backprop(self, prev_dy):

        # initializing derivative of trainable weights with zeros
        if self.trainable:
            dFWx, dFWh, dFWb = np.zeros(self.FWx.shape), np.zeros(self.FWh.shape), np.zeros(self.FWb.shape)
            dBWx, dBWh, dBWb = np.zeros(self.BWx.shape), np.zeros(self.BWh.shape), np.zeros(self.BWb.shape)


        # BackPropagation
        dx_f = []
        dx_b = []

        dh_ahead_f = np.zeros((self.inputs.shape[0], self.hidden_units_f))
        dh_ahead_b = np.zeros((self.inputs.shape[0], self.hidden_units_b))

        inputs_rev = self.inputs[:, ::-1, :]
        prev_dy_rev = prev_dy[:, ::-1, :]

        for t in reversed(range(self.input_shape[1])):

            # in forward direction
            dh_f = (prev_dy[:, t, :][:, :self.hidden_units_f] + dh_ahead_f) * (1 - self.prev_fh[t + 1] ** 2)

            if self.trainable:
                dFWx += self.inputs[:, t, :].T @ dh_f
                dFWh += self.prev_fh[t].T @ dh_f
                dFWb += np.sum(dh_f, axis = 0, keepdims = True)

            dh_ahead_f = dh_f @ self.FWh.weight.T
            dx_f.append(dh_f @ self.FWx.weight.T)
            

            # in backward propagation
            dh_b = (prev_dy_rev[:, t, :][:, self.hidden_units_f:] + dh_ahead_b) * (1 - self.prev_bh[t + 1] ** 2)

            if self.trainable:
                dBWx += inputs_rev[:, t, :].T @ dh_b
                dBWh += self.prev_bh[t].T @ dh_b
                dBWb += np.sum(dh_b, axis = 0, keepdims = True)

            dh_ahead_b = dh_b @ self.BWh.weight.T
            dx_b.append(dh_b @ self.BWx.weight.T)

        dx_f = np.transpose(np.array(dx_f[::-1]), axes = [1, 0, 2])
        dx_b = np.transpose(np.array(dx_b), axes = [1, 0, 2])
        dx = dx_f + dx_b

        # updating weights if the weights are trainable using optimizer
        if self.trainable:
            grad_pairs = [(self.FWx_opt, self.FWx, dFWx), (self.FWh_opt, self.FWh, dFWh), (self.FWb_opt, self.FWb, dFWb),
                          (self.BWx_opt, self.BWx, dBWx), (self.BWh_opt, self.BWh, dBWh), (self.BWb_opt, self.BWb, dBWb)]
            self._update_weights(grad_pairs)
            
        assert dx.shape == self.inputs.shape
        return dx

    @property
    def output_shape(self):
        batch, inp_seq, inp_dim = self.input_shape
        return (batch, inp_seq, self.hidden_units_f + self.hidden_units_b)


#####################################################################################################


# LSTM (Long Short Term Memory)
class LSTM(Layer):
    '''
        LSTM(Long Short Term Memory) Layer (Base Class: Layer)
            LSTM is another type of Recurrent Neural Network. It consists of
            diffrent gates and states, which helps in keeping track of long
            sequences. And, it also solves the problem of gradient vanishing.


        Args:
            hidden_units:
                - type: +ve integer
                - about: Defines output dimensionality. The number of hidden
                         neurons in the layer.

            return_sequences:
                - type: boolean
                - about: Decides whether to return hidden state from every time step.
                         If set as `False`, then would return the hidden state of the
                         last time step, otherwise return the hidden state from each
                         time step.
                - default: False

            **kwargs:
                    name:
                        - type: string
                        - about: Name of the layer can be defined.
                        - default: `LSTM`

                    trainable:
                        - type: boolean
                        - about: Defines if the weights of the layer is trainable.
                        - default: True

                    weight_initializer:
                        - type: string
                        - about: Appropriate Weight Initialization technique is defined.
                        - default: 'uniform'


        Input Shape:
            3-D tensor with shape (batch_size, sequence, input_dim)

        Output Shape:
            If return_sequences is set to True then,
                3-D tensor with shape (batch_size, sequence, hidden_units)
            otherwise,
                2-D tensor with shape (batch_size, hidden_units)
                
    '''
    def __init__(self, hidden_units: int, return_sequences: bool = False, **kwargs):
        self.hidden_units = hidden_units
        self.return_sequences = return_sequences
        
        self.input_shape = None
        self.all_weights = ()

        self.name = kwargs.get('name', self.__class__.__name__)
        self.trainable = kwargs.get('trainable', True)
        self.weight_initializer = kwargs.get('weight_initializer', 'uniform')

    def preset_optimizer(self, optimizer):
        self.Wfx_opt, self.Wfh_opt, self.Wfb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.Wix_opt, self.Wih_opt, self.Wib_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.Wgx_opt, self.Wgh_opt, self.Wgb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.Wox_opt, self.Woh_opt, self.Wob_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        

    def build(self, input_shape):
        self.input_shape = input_shape
        _, _, inp_dim = input_shape

        # forget gate
        self.Wfx = WeightsInitializer(shape = (inp_dim, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|ForgetGateWeight:Wfx')
        self.Wfh = WeightsInitializer(shape = (self.hidden_units, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|ForgetGateRecurrentWeight:Wfh')
        self.Wfb = WeightsInitializer(shape = (1, self.hidden_units),
                                      initializer = 'zeros', trainable = True,
                                      name = f'{self.name}|ForgetGateBias:Wfb')

        # input gate
        self.Wix = WeightsInitializer(shape = (inp_dim, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|InputGateWeight:Wix')
        self.Wih = WeightsInitializer(shape = (self.hidden_units, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|InputGateRecurrentWeight:Wih')
        self.Wib = WeightsInitializer(shape = (1, self.hidden_units),
                                      initializer = 'zeros', trainable = True,
                                      name = f'{self.name}|InputGateBias:Wib')

        # candidate state
        self.Wgx = WeightsInitializer(shape = (inp_dim, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|CandidateStateWeight:Wgx')
        self.Wgh = WeightsInitializer(shape = (self.hidden_units, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|CandidateStateRecurrentWeight:Wgh')
        self.Wgb = WeightsInitializer(shape = (1, self.hidden_units),
                                      initializer = 'zeros', trainable = True,
                                      name = f'{self.name}|CandidateStateBias:Wgb')

        # output gate
        self.Wox = WeightsInitializer(shape = (inp_dim, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|OutputGateWeight:Wox')
        self.Woh = WeightsInitializer(shape = (self.hidden_units, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|OutputGateRecurrentWeight:Woh')
        self.Wob = WeightsInitializer(shape = (1, self.hidden_units),
                                      initializer = 'zeros', trainable = True,
                                      name = f'{self.name}|OutputGateBias:Wob')

        self.all_weights = (self.Wfx, self.Wfh, self.Wfb, self.Wix, self.Wih, self.Wib,
                            self.Wgx, self.Wgh, self.Wgb, self.Wox, self.Woh, self.Wob)


    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def __tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        
    def __call__(self, inputs):
        self.inputs = inputs
        if not hasattr(self, 'Wob'):
            self.build(inputs.shape)

        # initializing hidden & candidate states with zeros
        c = np.zeros((inputs.shape[0], self.hidden_units))
        h = np.zeros((inputs.shape[0], self.hidden_units))

        # storing hidden & candidate states
        self.prev_c = {0: c}
        self.prev_h = {0: h}

        self.prev_f = []
        self.prev_i = []
        self.prev_g = []
        self.prev_o = []

        # Forward Propagation
        for t in range(self.input_shape[1]):
            f = self.__sigmoid(inputs[:, t, :] @ self.Wfx.weight + h @ self.Wfh.weight + self.Wfb.weight)
            i = self.__sigmoid(inputs[:, t, :] @ self.Wix.weight + h @ self.Wih.weight + self.Wib.weight)
            g = self.__tanh(inputs[:, t, :] @ self.Wgx.weight + h @ self.Wgh.weight + self.Wgb.weight)
            o = self.__sigmoid(inputs[:, t, :] @ self.Wox.weight + h @ self.Woh.weight + self.Wob.weight)

            c = f * c + i * g
            h = o * self.__tanh(c)

            self.prev_c[t + 1] = c
            self.prev_h[t + 1] = h

            self.prev_f.append(f)
            self.prev_i.append(i)
            self.prev_g.append(g)
            self.prev_o.append(o)

        # returns hidden states from each time step
        if self.return_sequences:
            out = np.transpose(np.array(list(self.prev_h.values())[1:]), axes = [1, 0, 2])
            assert out.shape[1:] == self.output_shape[1:]
            return out

        # return hidden state from last time step
        assert h.shape[-1] == self.output_shape[-1]
        return h


    def backprop(self, prev_dy):
        n = self.input_shape[1]

        # initializing derivative of trainable weights with zeros
        if self.trainable:
            dWfx, dWfh, dWfb = np.zeros(self.Wfx.shape), np.zeros(self.Wfh.shape), np.zeros(self.Wfb.shape)
            dWix, dWih, dWib = np.zeros(self.Wix.shape), np.zeros(self.Wih.shape), np.zeros(self.Wib.shape)
            dWgx, dWgh, dWgb = np.zeros(self.Wgx.shape), np.zeros(self.Wgh.shape), np.zeros(self.Wgb.shape)
            dWox, dWoh, dWob = np.zeros(self.Wox.shape), np.zeros(self.Woh.shape), np.zeros(self.Wob.shape)


        # BackPropagation
        if self.return_sequences:
            dh_ahead = np.zeros((self.inputs.shape[0], self.hidden_units))
            dc_ahead = np.zeros((self.inputs.shape[0], self.hidden_units))
        else:
            dh = prev_dy
            dc = dh * self.prev_o[n - 1] * (1 - self.__tanh(self.prev_c[n]) ** 2)

        dx = []

        for t in reversed(range(n)):

            if self.return_sequences:
                dh = prev_dy[:, t, :] + dh_ahead
                dc = dh * self.prev_o[t] * (1 - self.__tanh(self.prev_c[t + 1]) ** 2) + dc_ahead

            df = dc * self.prev_c[t] * self.prev_f[t] * (1 - self.prev_f[t])
            di = dc * self.prev_g[t] * self.prev_i[t] * (1 - self.prev_i[t])
            dg = dc * self.prev_i[t] * (1 - self.prev_g[t] ** 2)
            do = dh * self.__tanh(self.prev_c[t + 1]) * self.prev_o[t] * (1 - self.prev_o[t])

            if self.trainable:            
                dWfx += self.inputs[:, t, :].T @ df
                dWfh += self.prev_h[t].T @ df
                dWfb += np.sum(df, axis = 0, keepdims = True)

                dWix += self.inputs[:, t, :].T @ di
                dWih += self.prev_h[t].T @ di
                dWib += np.sum(di, axis = 0, keepdims = True)

                dWgx += self.inputs[:, t, :].T @ dg
                dWgh += self.prev_h[t].T @ dg
                dWgb += np.sum(dg, axis = 0, keepdims = True)

                dWox += self.inputs[:, t, :].T @ do
                dWoh += self.prev_h[t].T @ do
                dWob += np.sum(do, axis = 0, keepdims = True)


            if self.return_sequences:
                dh_ahead = do @ self.Woh.weight.T + dg @ self.Wgh.weight.T + di @ self.Wih.weight.T + df @ self.Wfh.weight.T
                dc_ahead = dc * self.prev_f[t]
            else:
                dc *= self.prev_f[t]
                dh = do @ self.Woh.weight.T + dg @ self.Wgh.weight.T + di @ self.Wih.weight.T + df @ self.Wfh.weight.T

            dx.append(do @ self.Wox.weight.T + dg @ self.Wgx.weight.T + di @ self.Wix.weight.T + df @ self.Wfx.weight.T)

        dx = np.transpose(np.array(dx[::-1]), axes = [1, 0, 2])
        
        # updating weights if the layer is trainable using an optimizer
        if self.trainable:
            grad_pairs = [(self.Wfx_opt, self.Wfx, dWfx), (self.Wfh_opt, self.Wfh, dWfh), (self.Wfb_opt, self.Wfb, dWfb),
                          (self.Wix_opt, self.Wix, dWix), (self.Wih_opt, self.Wih, dWih), (self.Wib_opt, self.Wib, dWib),
                          (self.Wgx_opt, self.Wgx, dWgx), (self.Wgh_opt, self.Wgh, dWgh), (self.Wgb_opt, self.Wgb, dWgb),
                          (self.Wox_opt, self.Wox, dWox), (self.Woh_opt, self.Woh, dWoh), (self.Wob_opt, self.Wob, dWob)]

            self._update_weights(grad_pairs)

        assert dx.shape == self.inputs.shape
        return dx


    @property
    def output_shape(self):
        batch, inp_seq, inp_dim = self.input_shape
        if self.return_sequences:
            return (batch, inp_seq, self.hidden_units)
        return (batch, self.hidden_units)


#####################################################################################################

    

# BidirectionalLSTM Layer
# Note: BidiretionalLSTM Layer could have also been implemented
# by calling two LSTM Layers in the below Layer, which could
# have reduced the work load.
class BidirectionalLSTM(Layer):
    '''
        BidirectionalLSTM Layer (Base Class: Layer)
            It is a Bidirectional LSTM Layer. Where inputs move from both
            directions through different gates & states, left-right & right-left,
            so as to get the information from the whole sentence.


        Args:
            hidden_units:
                - type: +ve integer / tuple of +ve integers
                - about: Defines output dimensionality. The number of hidden
                         neurons in the layer. If the input is tuple of shape 2,
                         then first integer defines the forward hidden units whereas,
                         the other defines the backward hidden units. If the input is
                         just an integer then, both forward and backward hidden units
                         have same number of neurons.

        **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `BidirectionalLSTM`

                trainable:
                    - type: boolean
                    - about: Defines if the weights of the layer is trainable.
                    - default: True

                weight_initializer:
                    - type: string
                    - about: Appropriate Weight Initialization technique is defined.
                    - default: 'uniform'

        Input Shape:
            3-D tensor of shape (batch_size, sequence, input_dim)

        Output Shape:
            If hidden units is integer, then
                3-D tensor of shape (batch_size, sequence, 2*hidden_units)
            otherwise, if hidden units is tuple, then,
                3-D tensor of shape (batch_size, sequence, hidden_units[0] + hidden_units[1])

        
    '''
    def __init__(self, hidden_units, **kwargs):
        self.input_shape = None
        self.all_weights = ()
        
        if type(hidden_units) == int:
            self.hidden_units_f = hidden_units
            self.hidden_units_b = hidden_units

        elif (type(hidden_units) == list) | (type(hidden_units) == tuple):
            self.hidden_units_f = hidden_units[0]
            self.hidden_units_b = hidden_units[1]

        else:
            raise Exception('Hidden Units must be `int`, `tuple` or `list`.')


        self.name = kwargs.get('name', self.__class__.__name__)
        self.trainable = kwargs.get('trainable', True)
        self.weight_initializer = kwargs.get('weight_initializer', 'uniform')

    def preset_optimizer(self, optimizer):
        self.FWfx_opt, self.FWfh_opt, self.FWfb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.FWix_opt, self.FWih_opt, self.FWib_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.FWgx_opt, self.FWgh_opt, self.FWgb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.FWox_opt, self.FWoh_opt, self.FWob_opt = copy(optimizer), copy(optimizer), copy(optimizer)

        self.BWfx_opt, self.BWfh_opt, self.BWfb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.BWix_opt, self.BWih_opt, self.BWib_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.BWgx_opt, self.BWgh_opt, self.BWgb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.BWox_opt, self.BWoh_opt, self.BWob_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        

    def build(self, input_shape):
        self.input_shape = input_shape
        _, _, inp_dim = input_shape

        ### Forward Weights
        # forget gate
        self.FWfx = WeightsInitializer(shape = (inp_dim, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|ForwardForgetGateWeight:FWfx')
        self.FWfh = WeightsInitializer(shape = (self.hidden_units_f, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|ForwardForgetGateRecurrentWeight:FWfh')
        self.FWfb = WeightsInitializer(shape = (1, self.hidden_units_f),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|ForwardForgetGateBias:FWfb')

        # input gate
        self.FWix = WeightsInitializer(shape = (inp_dim, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|ForwardInputGateWeight:FWix')
        self.FWih = WeightsInitializer(shape = (self.hidden_units_f, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|ForwardInputGateRecurrentWeight:FWih')
        self.FWib = WeightsInitializer(shape = (1, self.hidden_units_f),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|ForwardInputGateBias:FWib')

        # candidate gate
        self.FWgx = WeightsInitializer(shape = (inp_dim, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = 'BidirectionalLSTM|ForwardCandidateStateWeight:FWgx')
        self.FWgh = WeightsInitializer(shape = (self.hidden_units_f, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|ForwardCandidateStateRecurrentWeight:FWgh')
        self.FWgb = WeightsInitializer(shape = (1, self.hidden_units_f),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|ForwardCandidateStateBias:FWgb')

        # output gate
        self.FWox = WeightsInitializer(shape = (inp_dim, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|ForwardOutputGateWeight:FWox')
        self.FWoh = WeightsInitializer(shape = (self.hidden_units_f, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|ForwardOutputGateRecurrentWeight:FWoh')
        self.FWob = WeightsInitializer(shape = (1, self.hidden_units_f),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|ForwardOutputGateBias:FWob')



        ### Backward Weights
        # forget gate
        self.BWfx = WeightsInitializer(shape = (inp_dim, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardForgetGateWeight:BWfx')
        self.BWfh = WeightsInitializer(shape = (self.hidden_units_b, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardForgetGateRecurrentWeight:BWfh')
        self.BWfb = WeightsInitializer(shape = (1, self.hidden_units_b),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|BackwardForgetGateBias:BWfb')

        # input gate
        self.BWix = WeightsInitializer(shape = (inp_dim, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardInputGateWeight:BWix')
        self.BWih = WeightsInitializer(shape = (self.hidden_units_b, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardInputGateRecurrentWeight:BWih')
        self.BWib = WeightsInitializer(shape = (1, self.hidden_units_b),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|BackwardInputGateBias:BWib')

        # candidate gate
        self.BWgx = WeightsInitializer(shape = (inp_dim, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardCandidateStateWeight:BWgx')
        self.BWgh = WeightsInitializer(shape = (self.hidden_units_b, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardCandidateStateRecurrentWeight:BWgh')
        self.BWgb = WeightsInitializer(shape = (1, self.hidden_units_b),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|BackwardCandidateStateBias:BWgb')

        # output gate
        self.BWox = WeightsInitializer(shape = (inp_dim, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardOutputGateWeight:BWox')
        self.BWoh = WeightsInitializer(shape = (self.hidden_units_b, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardOutputGateRecurrentWeight:BWoh')
        self.BWob = WeightsInitializer(shape = (1, self.hidden_units_b),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|BackwardOutputGateBias:BWob')

        self.all_weights = (self.FWfx, self.FWfh, self.FWfb, self.FWix, self.FWih, self.FWib,
                            self.FWgx, self.FWgh, self.FWgb, self.FWox, self.FWoh, self.FWob,
                            self.BWfx, self.BWfh, self.BWfb, self.BWix, self.BWih, self.BWib,
                            self.BWgx, self.BWgh, self.BWgb, self.BWox, self.BWoh, self.BWob)
        

    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def __tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        

    def __call__(self, inputs):
        self.inputs = inputs
        if not hasattr(self, 'BWob'):
            self.build(inputs.shape)

        
        # initializing hidden & candidate state for both forward and backward direction
        fc, fh = np.zeros((inputs.shape[0], self.hidden_units_f)), np.zeros((inputs.shape[0], self.hidden_units_f))
        bc, bh = np.zeros((inputs.shape[0], self.hidden_units_b)), np.zeros((inputs.shape[0], self.hidden_units_b))

        self.prev_fc, self.prev_fh = {0: fc}, {0: fh}
        self.prev_bc, self.prev_bh = {0: bc}, {0: bh}

        self.prev_ff, self.prev_fi, self.prev_fg, self.prev_fo = [], [], [], []
        self.prev_bf, self.prev_bi, self.prev_bg, self.prev_bo = [], [], [], []

        inputs_rev = inputs[:, ::-1, :]

        # Forward Propagation
        for t in range(self.input_shape[1]):

            # in forward direction
            ff = self.__sigmoid(inputs[:, t, :] @ self.FWfx.weight + fh @ self.FWfh.weight + self.FWfb.weight)
            fi = self.__sigmoid(inputs[:, t, :] @ self.FWix.weight + fh @ self.FWih.weight + self.FWib.weight)
            fg = self.__tanh(inputs[:, t, :] @ self.FWgx.weight + fh @ self.FWgh.weight + self.FWgb.weight)
            fo = self.__sigmoid(inputs[:, t, :] @ self.FWox.weight + fh @ self.FWoh.weight + self.FWob.weight)

            fc = ff * fc + fi * fg
            fh = fo * self.__tanh(fc)

            self.prev_fc[t + 1] = fc
            self.prev_fh[t + 1] = fh

            self.prev_ff.append(ff)
            self.prev_fi.append(fi)
            self.prev_fg.append(fg)
            self.prev_fo.append(fo)

            # in backward direction
            bf = self.__sigmoid(inputs_rev[:, t, :] @ self.BWfx.weight + bh @ self.BWfh.weight + self.BWfb.weight)
            bi = self.__sigmoid(inputs_rev[:, t, :] @ self.BWix.weight + bh @ self.BWih.weight + self.BWib.weight)
            bg = self.__tanh(inputs_rev[:, t, :] @ self.BWgx.weight + bh @ self.BWgh.weight + self.BWgb.weight)
            bo = self.__sigmoid(inputs_rev[:, t, :] @ self.BWox.weight + bh @ self.BWoh.weight + self.BWob.weight)

            bc = bf * bc + bi * bg
            bh = bo * self.__tanh(bc)

            self.prev_bc[t + 1] = bc
            self.prev_bh[t + 1] = bh

            self.prev_bf.append(bf)
            self.prev_bi.append(bi)
            self.prev_bg.append(bg)
            self.prev_bo.append(bo)

        

        # concatenate hidden states from both forward & backward direction
        concat_h = []
        for f_, b_ in zip(list(self.prev_fh.values())[1:], list(self.prev_bh.values())[1:][::-1]):
            concat_h.append(np.concatenate([f_, b_], axis = 1))
            #assert concat_h[-1].shape == (self.inputs.shape[0], self.hidden_units_f + self.hidden_units_b)
        out = np.transpose(np.array(concat_h), axes = [1, 0, 2])

        assert out.shape[1:] == self.output_shape[1:]
        return out


    def backprop(self, prev_dy):
        #n = len(self.inputs)

        # initializing derivative of trainable weights in zeros
        if self.trainable:
            dFWfx, dFWfh, dFWfb = np.zeros(self.FWfx.shape), np.zeros(self.FWfh.shape), np.zeros(self.FWfb.shape)
            dFWix, dFWih, dFWib = np.zeros(self.FWix.shape), np.zeros(self.FWih.shape), np.zeros(self.FWib.shape)
            dFWgx, dFWgh, dFWgb = np.zeros(self.FWgx.shape), np.zeros(self.FWgh.shape), np.zeros(self.FWgb.shape)
            dFWox, dFWoh, dFWob = np.zeros(self.FWox.shape), np.zeros(self.FWoh.shape), np.zeros(self.FWob.shape)

            dBWfx, dBWfh, dBWfb = np.zeros(self.BWfx.shape), np.zeros(self.BWfh.shape), np.zeros(self.BWfb.shape)
            dBWix, dBWih, dBWib = np.zeros(self.BWix.shape), np.zeros(self.BWih.shape), np.zeros(self.BWib.shape)
            dBWgx, dBWgh, dBWgb = np.zeros(self.BWgx.shape), np.zeros(self.BWgh.shape), np.zeros(self.BWgb.shape)
            dBWox, dBWoh, dBWob = np.zeros(self.BWox.shape), np.zeros(self.BWoh.shape), np.zeros(self.BWob.shape)

        
        # BackPropagation
        dc_ahead_f, dh_ahead_f = np.zeros((self.inputs.shape[0], self.hidden_units_f)), np.zeros((self.inputs.shape[0], self.hidden_units_f))
        dc_ahead_b, dh_ahead_b = np.zeros((self.inputs.shape[0], self.hidden_units_b)), np.zeros((self.inputs.shape[0], self.hidden_units_b))

        dx_f, dx_b = [], []
        
        prev_dy_rev = prev_dy[:, ::-1, :]
        inp_rev = self.inputs[:, ::-1, :]

        for t in reversed(range(self.input_shape[1])):
            # in forward direction
            dfh = prev_dy[:, t, :][:, :self.hidden_units_f] + dh_ahead_f
            dfc = dfh * self.prev_fo[t] * (1 - self.__tanh(self.prev_fc[t + 1]) ** 2) + dc_ahead_f

            dff = dfc * self.prev_fc[t] * self.prev_ff[t] * (1 - self.prev_ff[t])
            dfi = dfc * self.prev_fg[t] * self.prev_fi[t] * (1 - self.prev_fi[t])
            dfg = dfc * self.prev_fi[t] * (1 - self.prev_fg[t] ** 2)
            dfo = dfh * self.__tanh(self.prev_fc[t + 1]) * self.prev_fo[t] * (1 - self.prev_fo[t])

            if self.trainable:
                dFWfx += self.inputs[:, t, :].T @ dff
                dFWfh += self.prev_fh[t].T @ dff
                dFWfb += np.sum(dff, axis = 0, keepdims = True)

                dFWix += self.inputs[:, t, :].T @ dfi
                dFWih += self.prev_fh[t].T @ dfi
                dFWib += np.sum(dfi, axis = 0, keepdims = True)

                dFWgx += self.inputs[:, t, :].T @ dfg
                dFWgh += self.prev_fh[t].T @ dfg
                dFWgb += np.sum(dfg, axis = 0, keepdims = True)
                
                dFWox += self.inputs[:, t, :].T @ dfo
                dFWoh += self.prev_fh[t].T @ dfo
                dFWob += np.sum(dfo, axis = 0, keepdims = True)

            dc_ahead_f = dfc * self.prev_ff[t]
            dh_ahead_f = dfo @ self.FWoh.weight.T + dfg @ self.FWgh.weight.T + dfi @ self.FWih.weight.T + dff @ self.FWfh.weight.T
            # dh_ahead_f = 0.25 * (dfo @ self.FWoh.weight.T + dfg @ self.FWgh.weight.T + dfi @ self.FWih.weight.T + dff @ self.FWfh.weight.T)
            
            dx_f.append(dfo @ self.FWox.weight.T + dfg @ self.FWgx.weight.T + dfi @ self.FWix.weight.T + dff @ self.FWfx.weight.T)
            
            # in backward direction
            dbh = prev_dy_rev[:, t, :][:, self.hidden_units_f:] + dh_ahead_b
            dbc = dbh * self.prev_bo[t] * (1 - self.__tanh(self.prev_bc[t + 1]) ** 2) + dc_ahead_b

            dbf = dbc * self.prev_bc[t] * self.prev_bf[t] * (1 - self.prev_bf[t])
            dbi = dbc * self.prev_bg[t] * self.prev_bi[t] * (1 - self.prev_bi[t])
            dbg = dbc * self.prev_bi[t] * (1 - self.prev_bg[t] ** 2)
            dbo = dbh * self.__tanh(self.prev_bc[t + 1]) * self.prev_bo[t] * (1 - self.prev_bo[t])

            if self.trainable:
                dBWfx += inp_rev[:, t, :].T @ dbf
                dBWfh += self.prev_bh[t].T @ dbf
                dBWfb += np.sum(dbf, axis = 0, keepdims = True)

                dBWix += inp_rev[:, t, :].T @ dbi
                dBWih += self.prev_bh[t].T @ dbi
                dBWib += np.sum(dbi, axis = 0, keepdims = True)

                dBWgx += inp_rev[:, t, :].T @ dbg
                dBWgh += self.prev_bh[t].T @ dbg
                dBWgb += np.sum(dbg, axis = 0, keepdims = True)

                dBWox += inp_rev[:, t, :].T @ dbo
                dBWoh += self.prev_bh[t].T @ dbo
                dBWob += np.sum(dbo, axis = 0, keepdims = True)

            dc_ahead_b = dbc * self.prev_bf[t]
            dh_ahead_b = dbo @ self.BWoh.weight.T + dbg @ self.BWgh.weight.T + dbi @ self.BWih.weight.T + dbf @ self.BWfh.weight.T
            # dh_ahead_b = 0.25 * (dbo @ self.BWoh.weight.T + dbg @ self.BWgh.weight.T + dbi @ self.BWih.weight.T + dbf @ self.BWfh.weight.T)

            dx_b.append(dbo @ self.BWox.weight.T + dbg @ self.BWgx.weight.T + dbi @ self.BWix.weight.T + dbf @ self.BWfx.weight.T)

        dx_f = np.transpose(np.array(dx_f[::-1]), axes = [1, 0, 2])
        dx_b = np.transpose(np.array(dx_b), axes = [1, 0, 2])

        dx = dx_f + dx_b
            
        # updating trainable weights using an optimizer
        if self.trainable:
            grad_pairs = [(self.FWfx_opt, self.FWfx, dFWfx), (self.FWfh_opt, self.FWfh, dFWfh), (self.FWfb_opt, self.FWfb, dFWfb),
                          (self.FWix_opt, self.FWix, dFWix), (self.FWih_opt, self.FWih, dFWih), (self.FWib_opt, self.FWib, dFWib),
                          (self.FWgx_opt, self.FWgx, dFWgx), (self.FWgh_opt, self.FWgh, dFWgh), (self.FWgb_opt, self.FWgb, dFWgb),
                          (self.FWox_opt, self.FWox, dFWox), (self.FWoh_opt, self.FWoh, dFWoh), (self.FWob_opt, self.FWob, dFWob),
                          (self.BWfx_opt, self.BWfx, dBWfx), (self.BWfh_opt, self.BWfh, dBWfh), (self.BWfb_opt, self.BWfb, dBWfb),
                          (self.BWix_opt, self.BWix, dBWix), (self.BWih_opt, self.BWih, dBWih), (self.BWib_opt, self.BWib, dBWib),
                          (self.BWgx_opt, self.BWgx, dBWgx), (self.BWgh_opt, self.BWgh, dBWgh), (self.BWgb_opt, self.BWgb, dBWgb),
                          (self.BWox_opt, self.BWox, dBWox), (self.BWoh_opt, self.BWoh, dBWoh), (self.BWob_opt, self.BWob, dBWob)]
            self._update_weights(grad_pairs)

        assert dx.shape == self.inputs.shape
        return dx


    @property
    def output_shape(self):
        batch, inp_seq, inp_dim = self.input_shape
        return (batch, inp_seq, self.hidden_units_f + self.hidden_units_b)


#####################################################################################################


# GRU(Gated Recurrent Unit) Layer
class GRU(Layer):
    '''
        GRU(Gated Recurrent Unit) Layer (Base Class: Layer)
            GRU is another type of Recurrent Neural Network. It consists of
            diffrent gates and states, which helps in keeping track of long
            sequences.


        Args:
            hidden_units:
                - type: +ve integer
                - about: Defines output dimensionality. The number of hidden
                         neurons in the layer.

            return_sequences:
                - type: boolean
                - about: Decides whether to return hidden state from every time step.
                         If set as `False`, then would return the hidden state of the
                         last time step, otherwise return the hidden state from each
                         time step.
                - default: False

            **kwargs:
                    name:
                        - type: string
                        - about: Name of the layer can be defined.
                        - default: `GRU`

                    trainable:
                        - type: boolean
                        - about: Defines if the weights of the layer is trainable.
                        - default: True

                    weight_initializer:
                        - type: string
                        - about: Appropriate Weight Initialization technique is defined.
                        - default: 'uniform'


        Input Shape:
            3-D tensor with shape (batch_size, sequence, input_dim)

        Output Shape:
            If return_sequences is set to True then,
                3-D tensor with shape (batch_size, sequence, hidden_units)
            otherwise,
                2-D tensor with shape (batch_size, hidden_units)
                
    '''
    def __init__(self, hidden_units: int, return_sequences: bool = False, **kwargs):
        self.hidden_units = hidden_units
        self.return_sequences = return_sequences
        
        self.input_shape = None
        self.all_weights = ()

        self.name = kwargs.get('name', self.__class__.__name__)
        self.trainable = kwargs.get('trainable', True)
        self.weight_initializer = kwargs.get('weight_initializer', 'uniform')

    def preset_optimizer(self, optimizer):
        self.Wrx_opt, self.Wrh_opt, self.Wrb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.Wzx_opt, self.Wzh_opt, self.Wzb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.Wgx_opt, self.Wgh_opt, self.Wgb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        

    def build(self, input_shape):
        self.input_shape = input_shape
        _, _, inp_dim = input_shape

        # reset gate
        self.Wrx = WeightsInitializer(shape = (inp_dim, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|ResetGateWeight:Wrx')
        self.Wrh = WeightsInitializer(shape = (self.hidden_units, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|ResetGateRecurrentWeight:Wrh')
        self.Wrb = WeightsInitializer(shape = (1, self.hidden_units),
                                      initializer = 'zeros', trainable = True,
                                      name = f'{self.name}|ResetGateBias:Wrb')

        # update gate
        self.Wzx = WeightsInitializer(shape = (inp_dim, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|UpdateGateWeight:Wzx')
        self.Wzh = WeightsInitializer(shape = (self.hidden_units, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|UpdateGateRecurrentWeight:Wzh')
        self.Wzb = WeightsInitializer(shape = (1, self.hidden_units),
                                      initializer = 'zeros', trainable = True,
                                      name = f'{self.name}|UpdateGateBias:Wzb')

        # candidate state
        self.Wgx = WeightsInitializer(shape = (inp_dim, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|CandidateStateWeight:Wgx')
        self.Wgh = WeightsInitializer(shape = (self.hidden_units, self.hidden_units),
                                      initializer = self.weight_initializer, trainable = True,
                                      name = f'{self.name}|CandidateStateRecurrentWeight:Wgh')
        self.Wgb = WeightsInitializer(shape = (1, self.hidden_units),
                                      initializer = 'zeros', trainable = True,
                                      name = f'{self.name}|CandidateStateBias:Wgb')

        self.all_weights = (self.Wrx, self.Wrh, self.Wrb, self.Wzx, self.Wzh, self.Wzb,
                            self.Wgx, self.Wgh, self.Wgb)


    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def __tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        
    def __call__(self, inputs):
        self.inputs = inputs
        if not hasattr(self, 'Wgb'):
            self.build(inputs.shape)

        # initializing hidden state
        h = np.zeros((inputs.shape[0], self.hidden_units))
        self.prev_h = {0: h}

        self.prev_r = []
        self.prev_z = []
        self.prev_g = []

        # Forward Propagation
        for t in range(self.input_shape[1]):
            r = self.__sigmoid(inputs[:, t, :] @ self.Wrx.weight + h @ self.Wrh.weight + self.Wrb.weight)
            z = self.__sigmoid(inputs[:, t, :] @ self.Wzx.weight + h @ self.Wzh.weight + self.Wzb.weight)
            g = self.__tanh(inputs[:, t, :] @ self.Wgx.weight + (h * r) @ self.Wgh.weight + self.Wgb.weight)

            h = (1 - z) * h + z * g

            self.prev_h[t + 1] = h

            self.prev_r.append(r)
            self.prev_z.append(z)
            self.prev_g.append(g)

        # returning hidden states from each time step
        if self.return_sequences:
            out = np.transpose(np.array(list(self.prev_h.values())[1:]), axes = [1, 0, 2])
            assert out.shape[1:] == self.output_shape[1:]
            return out

        # returning hidden state from last time step
        assert h.shape[-1] == self.output_shape[-1]
        return h


    def backprop(self, prev_dy):

        # initializing derivative of trainable weights as zeros
        if self.trainable:
            dWrx, dWrh, dWrb = np.zeros(self.Wrx.shape), np.zeros(self.Wrh.shape), np.zeros(self.Wrb.shape)
            dWzx, dWzh, dWzb = np.zeros(self.Wzx.shape), np.zeros(self.Wzh.shape), np.zeros(self.Wzb.shape)
            dWgx, dWgh, dWgb = np.zeros(self.Wgx.shape), np.zeros(self.Wgh.shape), np.zeros(self.Wgb.shape)


        # BackPropagation
        if self.return_sequences:
            dh_ahead = np.zeros((self.inputs.shape[0], self.hidden_units))
        else:
            dh = prev_dy

        dx = []
        for t in reversed(range(self.input_shape[1])):
            if self.return_sequences:
                dh = prev_dy[:, t, :] + dh_ahead

            dg = dh * self.prev_z[t] * (1 - self.prev_g[t] ** 2)
            dz = dh * (self.prev_g[t] - self.prev_h[t]) * self.prev_z[t] * (1 - self.prev_z[t])
            dr = dg * (self.prev_h[t] @ self.Wgh.weight) * self.prev_r[t] * (1 - self.prev_r[t])
            
            if self.trainable:
                dWgx += self.inputs[:, t, :].T @ dg
                dWgh += (self.prev_h[t] * self.prev_r[t]).T @ dg
                dWgb += np.sum(dg, axis = 0, keepdims = True)
                
                dWzx += self.inputs[:, t, :].T @ dz
                dWzh += self.prev_h[t].T @ dz
                dWzb += np.sum(dz, axis = 0, keepdims = True)

                dWrx += self.inputs[:, t, :].T @ dr
                dWrh += self.prev_h[t].T @ dr
                dWrb += np.sum(dr, axis = 0, keepdims = True)

            if self.return_sequences:
                dh_ahead = dh * (1 - self.prev_z[t])
            else:
                dh *= (1 - self.prev_z[t])

            dx.append(dg @ self.Wgx.weight.T + dz @ self.Wzx.weight.T + dr @ self.Wrx.weight.T)

        dx = np.transpose(np.array(dx[::-1]), axes = [1, 0, 2])

        
        # update trainable weights using an optimizer
        if self.trainable:        
            grad_pairs = [(self.Wgx_opt, self.Wgx, dWgx), (self.Wgh_opt, self.Wgh, dWgh), (self.Wgb_opt, self.Wgb, dWgb),
                          (self.Wzx_opt, self.Wzx, dWzx), (self.Wzh_opt, self.Wzh, dWzh), (self.Wzb_opt, self.Wzb, dWzb),
                          (self.Wrx_opt, self.Wrx, dWrx), (self.Wrh_opt, self.Wrh, dWrh), (self.Wrb_opt, self.Wrb, dWrb)]
            self._update_weights(grad_pairs)

        assert dx.shape == self.inputs.shape
        return dx

    @property
    def output_shape(self):
        batch, inp_seq, inp_dim = self.input_shape
        if self.return_sequences:
            return (batch, inp_seq, self.hidden_units)
        return (batch, self.hidden_units)


#####################################################################################################


# Bidirectional-GRU Layer
# Note: BidiretionalGRU Layer could have also been implemented
# by calling two GRU Layers in the below Layer, which could
# have reduced the work load.
class BidirectionalGRU(Layer):
    '''
        BidirectionalGRU Layer (Base Class: Layer)
            It is a Bidirectional GRU Layer. Where inputs move from both
            directions through different gates & states, left-right & right-left,
            so as to get the information from the whole sentence.


        Args:
            hidden_units:
                - type: +ve integer / tuple of +ve integers
                - about: Defines output dimensionality. The number of hidden
                         neurons in the layer. If the input is tuple of shape 2,
                         then first integer defines the forward hidden units whereas,
                         the other defines the backward hidden units. If the input is
                         just an integer then, both forward and backward hidden units
                         have same number of neurons.

        **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `BidirectionalGRU`

                trainable:
                    - type: boolean
                    - about: Defines if the weights of the layer is trainable.
                    - default: True

                weight_initializer:
                    - type: string
                    - about: Appropriate Weight Initialization technique is defined.
                    - default: 'uniform'

        Input Shape:
            3-D tensor of shape (batch_size, sequence, input_dim)

        Output Shape:
            If hidden units is integer, then
                3-D tensor of shape (batch_size, sequence, 2*hidden_units)
            otherwise, if hidden units is tuple, then,
                3-D tensor of shape (batch_size, sequence, hidden_units[0] + hidden_units[1])

        
    '''
    def __init__(self, hidden_units: int, **kwargs):
        self.input_shape = None
        self.all_weights = ()
        
        if type(hidden_units) == int:
            self.hidden_units_f = hidden_units
            self.hidden_units_b = hidden_units

        elif (type(hidden_units) == tuple) | (type(hidden_units) == list):
            self.hidden_units_f = hidden_units[0]
            self.hidden_units_b = hidden_units[1]

        else:
            raise Exception('Hidden units should be `int`, `list` or `tuple`.')

        self.name = kwargs.get('name', self.__class__.__name__)
        self.trainable = kwargs.get('trainable', True)
        self.weight_initializer = kwargs.get('weight_initializer', 'uniform')

    def preset_optimizer(self, optimizer):
        self.FWrx_opt, self.FWrh_opt, self.FWrb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.FWzx_opt, self.FWzh_opt, self.FWzb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.FWgx_opt, self.FWgh_opt, self.FWgb_opt = copy(optimizer), copy(optimizer), copy(optimizer)

        self.BWrx_opt, self.BWrh_opt, self.BWrb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.BWzx_opt, self.BWzh_opt, self.BWzb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        self.BWgx_opt, self.BWgh_opt, self.BWgb_opt = copy(optimizer), copy(optimizer), copy(optimizer)
        

    def build(self, input_shape):
        self.input_shape = input_shape
        _, _, inp_dim = input_shape
        
        ### Forward Weights
        # Reset Gate
        self.FWrx = WeightsInitializer(shape = (inp_dim, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|ForwardResetGateWeight:FWrx')
        self.FWrh = WeightsInitializer(shape = (self.hidden_units_f, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|ForwardResetGateRecurrentWeight:FWrh')
        self.FWrb = WeightsInitializer(shape = (1, self.hidden_units_f),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|ForwardResetGateBias:FWrb')

        # Update Gate
        self.FWzx = WeightsInitializer(shape = (inp_dim, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|ForwardUpdateGateWeight:FWzx')
        self.FWzh = WeightsInitializer(shape = (self.hidden_units_f, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|ForwardUpdateGateRecurrentWeight:FWzh')
        self.FWzb = WeightsInitializer(shape = (1, self.hidden_units_f),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|ForwardUpdateGateBias:FWzb')

        # Candidate State
        self.FWgx = WeightsInitializer(shape = (inp_dim, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|ForwardCandidateStateWeight:FWgx')
        self.FWgh = WeightsInitializer(shape = (self.hidden_units_f, self.hidden_units_f),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|ForwardCandidateStateRecurrentWeight:FWgh')
        self.FWgb = WeightsInitializer(shape = (1, self.hidden_units_f),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|ForwardCandidateStateBias:FWgb')

        ### Backward Weights
        # Reset Gate
        self.BWrx = WeightsInitializer(shape = (inp_dim, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardResetGateWeight:BWrx')
        self.BWrh = WeightsInitializer(shape = (self.hidden_units_b, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardResetGateRecurrentWeight:BWrh')
        self.BWrb = WeightsInitializer(shape = (1, self.hidden_units_b),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|BackwardResetGateBias:BWrb')

        # Update Gate
        self.BWzx = WeightsInitializer(shape = (inp_dim, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardUpdateGateWeight:BWzx')
        self.BWzh = WeightsInitializer(shape = (self.hidden_units_b, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardUpdateGateRecurrentWeight:BWzh')
        self.BWzb = WeightsInitializer(shape = (1, self.hidden_units_b),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|BackwardUpdateGateBias:BWzb')

        # Candidate State
        self.BWgx = WeightsInitializer(shape = (inp_dim, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardCandidateStateWeight:BWgx')
        self.BWgh = WeightsInitializer(shape = (self.hidden_units_b, self.hidden_units_b),
                                       initializer = self.weight_initializer, trainable = True,
                                       name = f'{self.name}|BackwardCandidateStateRecurrentWeight:BWgh')
        self.BWgb = WeightsInitializer(shape = (1, self.hidden_units_b),
                                       initializer = 'zeros', trainable = True,
                                       name = f'{self.name}|BackwardCandidateStateBias:BWgb')

        self.all_weights = (self.FWrx, self.FWrh, self.FWrb, self.FWzx, self.FWzh, self.FWzb,
                            self.FWgx, self.FWgh, self.FWgb, self.BWrx, self.BWrh, self.BWrb,
                            self.BWzx, self.BWzh, self.BWzb, self.BWgx, self.BWgh, self.BWgb)


    def __tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def __call__(self, inputs):
        self.inputs = inputs
        if not hasattr(self, 'BWgb'):
            self.build(inputs.shape)

        # initializing hidden states as zeros (forward & backward)
        fh = np.zeros((inputs.shape[0], self.hidden_units_f))
        bh = np.zeros((inputs.shape[0], self.hidden_units_b))

        self.prev_fh = {0: fh}
        self.prev_bh = {0: bh}

        self.prev_fr, self.prev_fz, self.prev_fg = [], [], []
        self.prev_br, self.prev_bz, self.prev_bg = [], [], []

        inputs_rev = inputs[:, ::-1, :]

        # Forward Propagation
        for t in range(self.input_shape[1]):
            
            # in forward direction
            fr = self.__sigmoid(inputs[:, t, :] @ self.FWrx.weight + fh @ self.FWrh.weight + self.FWrb.weight)
            fz = self.__sigmoid(inputs[:, t, :] @ self.FWzx.weight + fh @ self.FWzh.weight + self.FWzb.weight)
            fg = self.__tanh(inputs[:, t, :] @ self.FWgx.weight + (fh * fr) @ self.FWgh.weight + self.FWgb.weight)

            fh = (1 - fz) * fh + fz * fg

            self.prev_fh[t + 1] = fh

            self.prev_fr.append(fr)
            self.prev_fz.append(fz)
            self.prev_fg.append(fg)

            
            # in backward direction
            br = self.__sigmoid(inputs_rev[:, t, :] @ self.BWrx.weight + bh @ self.BWrh.weight + self.BWrb.weight)
            bz = self.__sigmoid(inputs_rev[:, t, :] @ self.BWzx.weight + bh @ self.BWzh.weight + self.BWzb.weight)
            bg = self.__tanh(inputs_rev[:, t, :] @ self.BWgx.weight + (bh * br) @ self.BWgh.weight + self.BWgb.weight)

            bh = (1 - bz) * bh + bz * bg

            self.prev_bh[t + 1] = bh

            self.prev_br.append(br)
            self.prev_bz.append(bz)
            self.prev_bg.append(bg)

        
        # concatenate hidden states of both forward & backward direction
        concat_h = []
        for f_, b_ in zip(list(self.prev_fh.values())[1:], list(self.prev_bh.values())[1:][::-1]):
            concat_h.append(np.concatenate([f_, b_], axis = 1))
            #assert concat_h[-1].shape == (self.inputs.shape[0], self.hidden_units_f + self.hidden_units_b)
        out = np.transpose(np.array(concat_h), axes = [1, 0, 2])

        assert out.shape[1:] == self.output_shape[1:]
        return out
    

    def backprop(self, prev_dy):

        # initializing derivative of trainable weights as zeros
        if self.trainable:
            dFWrx, dFWrh, dFWrb = np.zeros(self.FWrx.shape), np.zeros(self.FWrh.shape), np.zeros(self.FWrb.shape)
            dFWzx, dFWzh, dFWzb = np.zeros(self.FWzx.shape), np.zeros(self.FWzh.shape), np.zeros(self.FWzb.shape)
            dFWgx, dFWgh, dFWgb = np.zeros(self.FWgx.shape), np.zeros(self.FWgh.shape), np.zeros(self.FWgb.shape)

            dBWrx, dBWrh, dBWrb = np.zeros(self.BWrx.shape), np.zeros(self.BWrh.shape), np.zeros(self.BWrb.shape)
            dBWzx, dBWzh, dBWzb = np.zeros(self.BWzx.shape), np.zeros(self.BWzh.shape), np.zeros(self.BWzb.shape)
            dBWgx, dBWgh, dBWgb = np.zeros(self.BWgx.shape), np.zeros(self.BWgh.shape), np.zeros(self.BWgb.shape)


        # BackPropagation
        dh_ahead_f = np.zeros((self.inputs.shape[0], self.hidden_units_f))
        dh_ahead_b = np.zeros((self.inputs.shape[0], self.hidden_units_b))

        dx_f, dx_b = [], []

        prev_dy_rev = prev_dy[:, ::-1, :]
        inputs_rev = self.inputs[:, ::-1, :]
        for t in reversed(range(self.input_shape[1])):

            # in forward direction
            dfh = prev_dy[:, t, :][:, :self.hidden_units_f] + dh_ahead_f

            dfg = dfh * self.prev_fz[t] * (1 - self.prev_fg[t] ** 2)
            dfz = dfh * (self.prev_fg[t] - self.prev_fh[t]) * self.prev_fz[t] * (1 - self.prev_fz[t])
            dfr = dfg * (self.prev_fh[t] @ self.FWgh.weight) * self.prev_fr[t] * (1 - self.prev_fr[t])

            if self.trainable:
                dFWgx += self.inputs[:, t, :].T @ dfg
                dFWgh += (self.prev_fh[t] * self.prev_fr[t]).T @ dfg
                dFWgb += np.sum(dfg, axis = 0, keepdims = True)
                
                dFWzx += self.inputs[:, t, :].T @ dfz
                dFWzh += self.prev_fh[t].T @ dfz
                dFWzb += np.sum(dfz, axis = 0, keepdims = True)

                dFWrx += self.inputs[:, t, :].T @ dfr
                dFWrh += self.prev_fh[t].T @ dfr
                dFWrb += np.sum(dfr, axis = 0, keepdims = True)

            dh_ahead_f = dfh * (1 - self.prev_fz[t])

            dx_f.append(dfg @ self.FWgx.weight.T + dfz @ self.FWzx.weight.T + dfr @ self.FWrx.weight.T)


            # in backward direction
            dbh = prev_dy_rev[:, t, :][:, self.hidden_units_f:] + dh_ahead_b

            dbg = dbh * self.prev_bz[t] * (1 - self.prev_bg[t] ** 2)
            dbz = dbh * (self.prev_bg[t] - self.prev_bh[t]) * self.prev_bz[t] * (1 - self.prev_bz[t])
            dbr = dbg * (self.prev_bh[t] @ self.BWgh.weight) * self.prev_br[t] * (1 - self.prev_br[t])

            if self.trainable:
                dBWgx += inputs_rev[:, t, :].T @ dbg
                dBWgh += (self.prev_bh[t] * self.prev_br[t]).T @ dbg
                dBWgb += np.sum(dbg, axis = 0, keepdims = True)

                dBWzx += inputs_rev[:, t, :].T @ dbz
                dBWzh += self.prev_bh[t].T @ dbz
                dBWzb += np.sum(dbz, axis = 0, keepdims = True)

                dBWrx += inputs_rev[:, t, :].T @ dbr
                dBWrh += self.prev_bh[t].T @ dbr
                dBWrb += np.sum(dbr, axis = 0, keepdims = True)

            dh_ahead_b = dbh * (1 - self.prev_bz[t])

            dx_b.append(dbg @ self.BWgx.weight.T + dbz @ self.BWzx.weight.T + dbr @ self.BWrx.weight.T)

        dx_f = np.transpose(np.array(dx_f[::-1]), axes = [1, 0, 2])
        dx_b = np.transpose(np.array(dx_b), axes = [1, 0, 2])

        dx = dx_f + dx_b
                
        # updating weights using an optimizer
        if self.trainable:
            grad_pairs = [(self.FWrx_opt, self.FWrx, dFWrx), (self.FWrh_opt, self.FWrh, dFWrh), (self.FWrb_opt, self.FWrb, dFWrb),
                          (self.FWzx_opt, self.FWzx, dFWzx), (self.FWzh_opt, self.FWzh, dFWzh), (self.FWzb_opt, self.FWzb, dFWzb),
                          (self.FWgx_opt, self.FWgx, dFWgx), (self.FWgh_opt, self.FWgh, dFWgh), (self.FWgb_opt, self.FWgb, dFWgb),
                          (self.BWrx_opt, self.BWrx, dBWrx), (self.BWrh_opt, self.BWrh, dBWrh), (self.BWrb_opt, self.BWrb, dBWrb),
                          (self.BWzx_opt, self.BWzx, dBWzx), (self.BWzh_opt, self.BWzh, dBWzh), (self.BWzb_opt, self.BWzb, dBWzb),
                          (self.BWgx_opt, self.BWgx, dBWgx), (self.BWgh_opt, self.BWgh, dBWgh), (self.BWgb_opt, self.BWgb, dBWgb)]
            self._update_weights(grad_pairs)

        assert dx.shape == self.inputs.shape
        return dx

    @property
    def output_shape(self):
        batch, inp_seq, inp_dim = self.input_shape
        return (batch, inp_seq, self.hidden_units_f + self.hidden_units_b)




# ==============================================================================================================================
#    CNN (Convolutional Neural Network)
# ==============================================================================================================================


# Convolutional 2D Layer
class Conv2D(Layer, Pad2D):
    '''
        Convolutional 2D Layer (Base Class: Layer, Pad2D)
            It does convolutional operation over images.


        Args:
            filters:
                - type: int
                - about: Output filters/feature maps are defined here.

            kernel_size:
                - type: tuple
                - about: It defines the kernel_size that will be used for convolution
                         operation. It consists of two values defining height and width
                         of the window respectively.

            strides:
                - type: tuple
                - about: It defines the amount of strides of convolution along the
                         height & width.
                - default: (1, 1)

            padding:
                - type: string/int/tuple
                - about: It defines the amount or type of padding that need
                         to be applied. If it is integer then, it simply defines
                         pad_value in top, bottom, left, right; whereas if it is
                         tuple consisting integers, then it defines padding in
                         height and width; and if it is tuple consisting tuples,
                         then, those denotes the amount of padding in height and
                         width, in top, bottom, left and right.

                         If it is string, then, there are two possibilities,
                         one being `valid`, where no padding is used. And other
                         being `same`, where padding is used in such a way that,
                         output shape is same as input (when stride is 1, otherwise
                         output_shape = input_shape/2, for height and width).
                - default: 'valid'

            activation:
                - type: string
                - about: defines appropriate activation function name.
                - default: None

            **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `Conv2D`

                trainable:
                    - type: boolean
                    - about: Defines if the weights of the layer is trainable.
                    - default: True

                weight_initializer:
                    - type: string
                    - about: Appropriate Weight Initialization technique is defined.
                    - default: 'uniform'

        Input Shape:
            4-D tensor of shape (batch_size, inp_height, inp_width, inp_feature_maps)

        Output Shape:
            4-D tensor of shape (batch_size, new_h, new_w, filters)
            where,
                new_h = int((inp_height - kernel_size[0] 2*pad_h) / strides[0]) + 1
                new_w = int((inp_width - kernel_size[1] 2*pad_w) / strides[1]) + 1

                where,
                    pad_h = padding along height
                    pad_w = padding along width

                
    '''
    def __init__(self, filters: int, kernel_size: tuple, strides: tuple = (1, 1),
                 padding: str = 'valid', activation: str = None, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        
        self.input_shape = None
        self.all_weights = ()

        self.name = kwargs.get('name', self.__class__.__name__)
        self.trainable = kwargs.get('trainable', True)
        self.weight_initializer = kwargs.get('weight_initializer', 'uniform')

    def preset_optimizer(self, optimizer):
        self.kernel_opt = copy(optimizer)
        self.bias_opt = copy(optimizer)

    def build(self, input_shape):
        self.input_shape = input_shape
        _, inp_h, inp_w, inp_chn = input_shape
        self.kernel = WeightsInitializer(shape = (self.kernel_size[0], self.kernel_size[1], inp_chn, self.filters),
                                         trainable = True, initializer = self.weight_initializer,
                                         name = f'{self.name}|Kernel:kernel')
        self.bias = WeightsInitializer(shape = (1, self.filters), trainable = True,
                                       initializer = 'zeros', name = f'{self.name}|Bias:bias')

        self.all_weights = (self.kernel, self.bias)


    @property
    def __im2col(self):
        batch_size, inp_h, inp_w, inp_chn = self.input_shape
        _, out_h, out_w, _ = self.output_shape

        # fancy indices for convolutional operation.
        h = np.tile(np.tile(np.tile(np.repeat(np.arange(self.kernel_size[0]), self.kernel_size[1]), inp_chn), out_w)\
                    .reshape(out_w, -1).T, out_h).T \
                    + np.repeat(np.arange(0, out_h * self.strides[0], self.strides[0]), np.prod(self.kernel_size) * inp_chn * out_w)\
                    .reshape(-1, np.prod(self.kernel_size) * inp_chn)

        w = np.tile(np.tile(np.tile(np.tile(np.arange(self.kernel_size[1]), self.kernel_size[0]), inp_chn), out_w).reshape(out_w, -1).T, out_h)\
            + np.tile(np.repeat(np.arange(0, out_w * self.strides[1], self.strides[1]), inp_chn * np.prod(self.kernel_size)).reshape(out_w, -1).T, out_h)

        c = np.repeat(np.arange(inp_chn), np.prod(self.kernel_size)).reshape(1, -1)

        return h, w, c

    def __call__(self, inputs):
        self.inputs = inputs
        if not hasattr(self, 'bias'):
            self.build(inputs.shape)

        batch_size, inp_h, inp_w, inp_chn = self.input_shape

        _, out_h, out_w, _ = self.output_shape
        pad_h, pad_w = self._padding(self.kernel_size, self.strides, self.padding)

        # Forward Propagation
        self.padded_inp = np.pad(self.inputs, ((0, 0), pad_h, pad_w, (0, 0)), mode = 'constant')

        # leveraging the feature of fancy indexing
        # of numpy for doing convolutional operation
        # instead of using for loops.
        # We use im2col method.
        # Know more about it in the CS231n course.
        h, w, c = self.__im2col

        self.inp_col = self.padded_inp[:, h, w.T, c]
        self.kernel_col = self.kernel.weight.reshape(np.prod(self.kernel_size), inp_chn, self.filters).transpose(1, 0, 2).reshape(-1, self.filters)
  
        out = (self.inp_col @ self.kernel_col).reshape(self.inputs.shape[0], out_h, out_w, self.filters)

        # applying activation function if any
        self.act = Activation(self.activation)
        out = self.act(out) if self.activation is not None else out

        assert out.shape[1:] == self.output_shape[1:]
        return out

    def backprop(self, prev_dy):
        _, out_h, out_w, _ = self.output_shape
        pad_h, pad_w = self._padding(self.kernel_size, self.strides, self.padding)
        batch, inp_h, inp_w, inp_chn = self.input_shape

        # BackPropagation
        prev_dy = self.act(prev_dy) if self.activation is not None else prev_dy

        if self.trainable:

            # using fancy for calculating derivative of the
            # kernel weight and bias.
            i = np.tile(np.tile(np.repeat(np.arange(0, out_h * self.strides[0], self.strides[0]), out_w).reshape(-1, 1), self.kernel_size[1]), self.kernel_size[0]).T\
                + np.repeat(np.arange(self.kernel_size[0]), self.kernel_size[1] * out_w * out_h).reshape(-1, out_h * out_w)

            j = np.tile(np.tile(np.tile(np.arange(0, out_w * self.strides[1], self.strides[1]), out_h).reshape(-1, 1), self.kernel_size[1]), self.kernel_size[0])\
                 + np.tile(np.repeat(np.arange(self.kernel_size[1]), out_w * out_h).reshape(-1, out_h * out_w).T, self.kernel_size[1])


            d_inp = self.padded_inp[:, i.T, j, :].reshape(-1, np.prod(self.kernel_size), inp_chn).transpose(1, 2, 0)
            d_prev = prev_dy.reshape(self.inputs.shape[0], -1, self.filters).reshape(-1, self.filters)

            d_kernel = (d_inp @ d_prev).reshape(self.kernel_size[0], self.kernel_size[1], inp_chn, self.filters)
            d_bias = np.sum(d_prev, axis = 0)[np.newaxis, :]

            grad_pairs = [(self.kernel_opt, self.kernel, d_kernel),
                          (self.bias_opt, self.bias, d_bias)]
            self._update_weights(grad_pairs)


        dx = np.zeros_like(self.padded_inp)
        z = prev_dy.reshape(self.inputs.shape[0], -1, self.filters) @ self.kernel_col.T
        h, w, c = self.__im2col
        
        np.add.at(dx, (slice(None), h, w.T, c), z)


        if (pad_h[1] > 0) & (pad_w[1] > 0):
            out =  dx[:, pad_h[0]:-pad_h[1], pad_w[0]:-pad_w[1], :]
        elif (pad_h[1] <= 0) & (pad_w[1] <= 0):
            out = dx
        elif (pad_h[1] > 0) & (pad_w[1] <= 0):
            out = dx[:, pad_h[0]:-pad_h[1], :, :]
        elif (pad_h[1] <= 0) & (pad_w[1] > 0):
            out = dx[:, :, pad_w[0]:-pad_w[1], :]

        
        assert out.shape == self.inputs.shape
        return out

    @property
    def output_shape(self):
        pad_h, pad_w = self._padding(self.kernel_size, self.strides, self.padding)
        batch, inp_h, inp_w, inp_chn = self.input_shape
        
        out_h = np.int((inp_h - self.kernel_size[0] + np.sum(pad_h)) / self.strides[0]) + 1
        out_w = np.int((inp_w - self.kernel_size[1] + np.sum(pad_w)) / self.strides[1]) + 1

        return (batch, out_h, out_w, self.filters)


#####################################################################################################


# Flatten Layer
class Flatten(Layer):
    '''
        Flatten Layer (Base Class: Layer)
            This layer converts any N-Dimensional tensor into 2-Dimensional
            tensor of shape (batch_size, n).


        Args:
            **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `Flatten`

        Input Shape:
            N-D tensor of shape (batch_size, ...)

        Output Shape:
            2-D tensor of shape (batch_size, n)
            
            
    '''
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', self.__class__.__name__)
        
        self.input_shape = None
        self.all_weights = ()

    def __call__(self, inputs):
        self.inputs = inputs
        if self.input_shape is None:
            self.input_shape = inputs.shape

        # Flattening (or reshaping input) in forward propagation
        out = inputs.reshape(self.inputs.shape[0], -1)

        assert out.shape[1:] == self.output_shape[1:]
        return out

    def backprop(self, prev_dy):
        # Reshaping gradients into input shape in backward propagation
        out = prev_dy.reshape(self.inputs.shape)

        assert out.shape == self.inputs.shape
        return out

    @property
    def output_shape(self):
        return (self.input_shape[0], np.prod(self.input_shape[1:]))
    

#####################################################################################################


# Max Pooling 2D Layer
class MaxPool2D(Layer, Pad2D):
    '''
        MaxPool2D Layer (Base Class: Layer, Pad2D)
            This layer takes the maximum value from the each window
            in the image of size defined in the layer with some strides
            along height and width.


        Args:
            pool_size:
                - type: tuple
                - about: It defines the window size along height and width.
                - default: (2, 2)

            strides:
                - type: tuple
                - about: It defines the amount of strides that needs to be used
                         along height and width by the windows.
                - default: (2, 2)

            padding:
                - type: string
                - about: It defines the amount or type of padding that need
                         to be applied. If it is integer then, it simply defines
                         pad_value in top, bottom, left, right; whereas if it is
                         tuple consisting integers, then it defines padding in
                         height and width; and if it is tuple consisting tuples,
                         then, those denotes the amount of padding in height and
                         width, in top, bottom, left and right.

                         If it is string, then, there are two possibilities,
                         one being `valid`, where no padding is used. And other
                         being `same`, where padding is used in such a way that,
                         output shape is same as input (when stride is 1, otherwise
                         output_shape = input_shape/2, for height and width).
                - default: 'same'

            **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `MaxPool2D`

        Input Shape:
            4-D tensor of shape (batch_size, inp_height, inp_width, inp_feature_maps)

        Output Shape:
            4-D tensor of shape (batch_size, new_h, new_w, inp_feature_maps)
                        
                
    '''
    def __init__(self, pool_size: tuple = (2, 2), strides: tuple = (2, 2),
                 padding: str = 'same', **kwargs):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        
        self.input_shape = None
        self.all_weights = ()

        self.name = kwargs.get('name', self.__class__.__name__)
    

    def __im2col(self, get_indices = False):
        pad_h, pad_w = self._padding(self.pool_size, self.strides, self.padding)
        self.pad_inp = np.pad(self.inputs, ((0, 0), pad_h, pad_w, (0, 0)), mode = 'constant')
        self.out_h = np.int((self.input_shape[1] - self.pool_size[0] + np.sum(pad_h))/self.strides[0]) + 1
        self.out_w = np.int((self.input_shape[2] - self.pool_size[1] + np.sum(pad_w))/self.strides[1]) + 1

        # leveraging the feature of fancy indexing
        # of numpy for doing maxpool operation, instead
        # of using for loops for each window region across
        # height & width.
        h = np.tile(np.tile(np.repeat(np.arange(self.pool_size[0]), self.pool_size[1]), self.out_w).reshape(-1, np.prod(self.pool_size)).T, self.out_h)
        h+= np.repeat(np.arange(0, self.out_h * self.strides[0], self.strides[0]), np.prod(self.pool_size) * self.out_w).reshape(-1, np.prod(self.pool_size)).T

        w = np.tile(np.tile(np.tile(np.arange(self.pool_size[1]), self.pool_size[0]), self.out_w).reshape(-1, np.prod(self.pool_size)).T, self.out_h)
        w += np.tile(np.repeat(np.arange(0, self.out_w * self.strides[1], self.strides[1]), np.prod(self.pool_size)).reshape(-1, np.prod(self.pool_size)).T, self.out_h)

        if get_indices:
            return h.T, w.T
        return self.pad_inp[:, h.T, w.T, :]


    def __call__(self, inputs):
        self.inputs = inputs
        if self.input_shape is None:
            self.input_shape = inputs.shape

        # forward propagation
        self.col_inp = self.__im2col()

        out = np.max(self.col_inp, axis = -2).reshape(self.inputs.shape[0], self.out_h, self.out_w, self.input_shape[-1])

        assert out.shape[1:] == self.output_shape[1:]
        return out

    def backprop(self, prev_dy):
        labels = np.prod(self.pool_size)
        batch_size, _, _, f_maps = self.input_shape

        # BackPropagation
        one_hot_classes = np.argmax(self.col_inp, axis = -2)[:, :, np.newaxis, :]\
                          .reshape(-1, 1, f_maps).transpose(0, 2, 1).reshape(-1, 1)[:, 0]
        one_hot_tar = np.eye(labels)[one_hot_classes]

        one_hot = one_hot_tar.reshape(-1, f_maps, labels).transpose(0, 2, 1).reshape(self.inputs.shape[0], -1, labels, f_maps)


        h_, w_ = self.__im2col(get_indices = True)
        dx = np.zeros_like(self.pad_inp)
        np.add.at(dx, (slice(None), h_, w_, slice(None)), one_hot)

        pad_h, pad_w = self._padding(self.pool_size, self.strides, self.padding)
        if (pad_h[1]>0) & (pad_w[1]>0):
            dx = dx[:, pad_h[0]:-pad_h[1], pad_w[0]:-pad_w[1], :]
        elif (pad_h[1] <= 0) & (pad_w[1] > 0):
            dx = dx[:, :, pad_w[0]:-pad_w[1], :]
        elif (pad_h[1] > 0) & (pad_w[1] <= 0):
            dx = dx[:, pad_h[0]:-pad_h[1], :, :]
        elif (pad_h[1] <= 0) & (pad_w[1] <= 0):
            pass

        z = np.repeat(np.repeat(prev_dy, self.strides[0], axis = 1), self.strides[1], axis = 2)
        z = z[:, :dx.shape[1], :dx.shape[2], :]

        out = z * dx

        assert out.shape == self.inputs.shape
        return out

    @property
    def output_shape(self):
        batch, inp_h, inp_w, inp_chn = self.input_shape
        if isinstance(self.padding, str):
            if self.padding == 'valid':
                out_h = np.int(np.floor((inp_h - self.pool_size[0])/self.strides[0]) + 1)
                out_w = np.int(np.floor((inp_w - self.pool_size[1])/self.strides[1]) + 1)

            elif self.padding == 'same':
                out_h = np.int(np.floor((inp_h - 1)/self.strides[0]) + 1)
                out_w = np.int(np.floor((inp_w - 1)/self.strides[1]) + 1)

        elif (isinstance(self.padding, tuple) | isinstance(self.padding, int)):
            pad_h, pad_w = self._padding(self.pool_size, self.strides, padding = self.padding)
            out_h = np.int((inp_h - self.pool_size[0] + np.sum(pad_h))/self.strides[0]) + 1
            out_w = np.int((inp_w - self.pool_size[1] + np.sum(pad_w))/self.strides[1]) + 1

        return (batch, out_h, out_w, inp_chn)


#####################################################################################################


# Average Pooling 2D Layer
class AveragePool2D(Layer, Pad2D):
    '''
        AveragePool2D Layer (Base Class: Layer, Pad2D)
            This layer takes the average of values from each window
            in the image of size defined in the layer with some strides
            along height and width.


        Args:
            pool_size:
                - type: tuple
                - about: It defines the window size along height and width.
                - default: (2, 2)

            strides:
                - type: tuple
                - about: It defines the amount of strides that needs to be used
                         along height and width by the windows.
                - default: (2, 2)

            padding:
                - type: string
                - about: It defines the amount or type of padding that need
                         to be applied. If it is integer then, it simply defines
                         pad_value in top, bottom, left, right; whereas if it is
                         tuple consisting integers, then it defines padding in
                         height and width; and if it is tuple consisting tuples,
                         then, those denotes the amount of padding in height and
                         width, in top, bottom, left and right.

                         If it is string, then, there are two possibilities,
                         one being `valid`, where no padding is used. And other
                         being `same`, where padding is used in such a way that,
                         output shape is same as input (when stride is 1, otherwise
                         output_shape = input_shape/2, for height and width).
                - default: 'same'

            **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `MaxPool2D`

        Input Shape:
            4-D tensor of shape (batch_size, inp_height, inp_width, inp_feature_maps)

        Output Shape:
            4-D tensor of shape (batch_size, new_h, new_w, inp_feature_maps)
                               
    
    '''
    def __init__(self, pool_size: tuple = (2, 2), strides: tuple = (2, 2),
                 padding: str = 'same', **kwargs):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        
        self.input_shape = None
        self.all_weights = ()

        self.name = kwargs.get('name', self.__class__.__name__)


    def __im2col(self, get_indices = False):
        pad_h, pad_w = self._padding(self.pool_size, self.strides, self.padding)

        self.pad_inp = np.pad(self.inputs, ((0, 0), pad_h, pad_w, (0, 0)), mode = 'constant')
        self.out_h = np.int((self.input_shape[1] - self.pool_size[0] + np.sum(pad_h))/self.strides[0]) + 1
        self.out_w = np.int((self.input_shape[2] - self.pool_size[1] + np.sum(pad_w))/self.strides[1]) + 1

        # leveraging the feature of fancy indexing
        # of numpy for doing maxpool operation, instead
        # of using for loops for each window region across
        # height & width.
        h = np.tile(np.tile(np.repeat(np.arange(self.pool_size[0]), self.pool_size[1]), self.out_w).reshape(-1, np.prod(self.pool_size)).T, self.out_h)
        h+= np.repeat(np.arange(0, self.out_h * self.strides[0], self.strides[0]), np.prod(self.pool_size) * self.out_w).reshape(-1, np.prod(self.pool_size)).T

        w = np.tile(np.tile(np.tile(np.arange(self.pool_size[1]), self.pool_size[0]), self.out_w).reshape(-1, np.prod(self.pool_size)).T, self.out_h)
        w += np.tile(np.repeat(np.arange(0, self.out_w * self.strides[1], self.strides[1]), np.prod(self.pool_size)).reshape(-1, np.prod(self.pool_size)).T, self.out_h)

        if get_indices:
            return h.T, w.T
        return self.pad_inp[:, h.T, w.T, :]

    def __call__(self, inputs):
        self.inputs = inputs

        if self.input_shape is None:
            self.input_shape = inputs.shape

        # ForwardPropagation
        self.col_inp = self.__im2col()

        out = np.mean(self.col_inp, axis = -2).reshape(self.inputs.shape[0], self.out_h, self.out_w, self.input_shape[-1])

        assert out.shape[1:] == self.output_shape[1:]
        return out

    def backprop(self, prev_dy):

        z = np.zeros_like(self.inputs)
        ph, pw = self.pool_size

        _, inp_h, inp_w, _ = self.inputs.shape

        # BackPropagation
        z = np.repeat(np.repeat(prev_dy, self.strides[0], axis = 1), self.strides[1], axis = 2)
        z = z[:, :inp_h, :inp_w, :]
        
        assert z.shape == self.inputs.shape
        return z

    @property
    def output_shape(self):
        batch, inp_h, inp_w, inp_chn = self.input_shape
        if isinstance(self.padding, str):
            if self.padding == 'valid':
                out_h = np.int(np.floor((inp_h - self.pool_size[0])/self.strides[0]) + 1)
                out_w = np.int(np.floor((inp_w - self.pool_size[1])/self.strides[1]) + 1)

            elif self.padding == 'same':
                out_h = np.int(np.floor((inp_h - 1)/self.strides[0]) + 1)
                out_w = np.int(np.floor((inp_w - 1)/self.strides[1]) + 1)

        elif (isinstance(self.padding, tuple) | isinstance(self.padding, int)):
            pad_h, pad_w = self._padding(padding = self.padding)
            out_h = np.int((inp_h - self.pool_size[0] + np.sum(pad_h))/self.strides[0]) + 1
            out_w = np.int((inp_w - self.pool_size[1] + np.sum(pad_w))/self.strides[1]) + 1

        return (batch, out_h, out_w, inp_chn)
    
#####################################################################################################
    

# Constant Padding 2D Layer
class ConstantPad2D(Layer, Pad2D):
    '''
        ConstantPad2D Layer (Base Class: Layer, Pad2D)
            This layer apply constant value (padding) in top, bottom,
            left and right of an image.


        Args:
            padding:
                - type: tuple
                - about: It is a tuple with two values,
                         defines the padding amount across height and width.

            padding_value:
                - type: int
                - about: It defines the constant value that needed to be applied
                         across height and width.
                - default: 0

            **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `ConstantPad2D`

        Input Shape:
            4-D tensor with shape (batch_size, inp_height, inp_width, inp_feature_maps)

        Output Shape:
            4-D tensor with shape (batch_size, inp_height + pad_h, inp_width + pad_w, inp_feature_maps)
            where,
                pad_h = total padding applied along height
                pad_w = total padding applied along width
                
                    
    '''
    def __init__(self, padding: tuple, padding_value = 0, **kwargs):
        self.padding = padding
        self.padding_value = padding_value
        
        self.input_shape = None
        self.all_weights = ()

        self.pad_h, self.pad_w = self._padding(padding = padding)

        self.name = kwargs.get('name', self.__class__.__name__)


    def __call__(self, inputs):
        self.inputs = inputs
        if self.input_shape is None:
            self.input_shape = inputs.shape

        # forward propagation
        out = np.pad(inputs, ((0, 0), self.pad_h, self.pad_w, (0, 0)),
                     mode = 'constant', constant_values = self.padding_value)

        assert self.output_shape[1:] == out.shape[1:]
        return out
        

    def backprop(self, prev_dy):
        b, h, w, c = self.input_shape
        
        # backward propagation
        out = prev_dy[:, self.pad_h[0]:h+self.pad_h[1], self.pad_w[0]:w+self.pad_w[1], :]
        assert out.shape == self.inputs.shape
        return out


    @property
    def output_shape(self):
        batch, inp_h, inp_w, inp_chn = self.input_shape
        return (batch, inp_h+np.sum(self.pad_h), inp_w+np.sum(self.pad_w), inp_chn)


#####################################################################################################


# Upsampling 2D Layer
class UpSampling2D(Layer):
    '''
        UpSampling2D Layer (Base Class: Layer)
            It uses nearest neighbor technique to upsample the image.
            The `size` parameter helps in defining, how much to upsample
            along height and width of the image.


        Args:
            size:
                - type: tuple
                - about: defines the amount of upsampling to be applied along
                         height and width.
                - default: (2, 2)

            **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `UpSampling2D`

        Input Shape:
            4-D tensor with shape (batch_size, inp_height, inp_width, inp_feature_maps)

        Output Shape:
            4-D tensor with shape (batch_size, size[0]*inp_height, size[1]*inp_width, inp_feature_maps)

    '''
    def __init__(self, size = (2, 2), **kwargs):
        self.size = size
        
        self.input_shape = None
        self.all_weights = ()

        self.name = kwargs.get('name', self.__class__.__name__)

    def __call__(self, inputs):
        self.inputs = inputs
        if self.input_shape is None:
            self.input_shape = inputs.shape

        # forward propagation
        out = np.repeat(np.repeat(self.inputs, self.size[0], axis = 1),
                        self.size[1], axis = 2)

        assert out.shape[1:] == self.output_shape[1:] 
        return out

    def backprop(self, prev_dy):
        # backward propagation
        out = prev_dy[:, ::self.size[0], ::self.size[1], :]
        assert self.inputs.shape == out.shape
        return out

    @property
    def output_shape(self):
        batch, inp_h, inp_w, inp_chn = self.input_shape
        return (batch, inp_h * self.size[0], inp_w * self.size[1], inp_chn)


#####################################################################################################
    

# Convolution 2D Transpose Layer
class Conv2DTranspose(Layer, Pad2D):
    '''
        Conv2DTranspose Layer (Base Class: Layer, Pad2D)
            This is Convolutional Transpose Layer.


        Args:
            filters:
                - type: int
                - about: Output filters/feature maps are defined here.

            kernel_size:
                - type: tuple
                - about: It defines the kernel_size that will be used for convolution
                         operation. It consists of two values defining height and width
                         of the window respectively.

            strides:
                - type: tuple
                - about: It defines the amount of strides of convolution along the
                         height & width.
                - default: (1, 1)

            padding:
                - type: string/int/tuple
                - about: It defines the amount or type of padding that need
                         to be applied. If it is integer then, it simply defines
                         pad_value in top, bottom, left, right; whereas if it is
                         tuple consisting integers, then it defines padding in
                         height and width; and if it is tuple consisting tuples,
                         then, those denotes the amount of padding in height and
                         width, in top, bottom, left and right.

                         If it is string, then, there are two possibilities,
                         one being `valid`, where no padding is used. And other
                         being `same`, where padding is used in such a way that,
                         output shape is same as input (when stride is 1, otherwise
                         output_shape = input_shape/2, for height and width).
                - default: 'valid'

            activation:
                - type: string
                - about: defines appropriate activation function name.
                - default: None

            **kwargs:
                name:
                    - type: string
                    - about: Name of the layer can be defined.
                    - default: `Conv2DTranspose`

                trainable:
                    - type: boolean
                    - about: Defines if the weights of the layer is trainable.
                    - default: True

                weight_initializer:
                    - type: string
                    - about: Appropriate Weight Initialization technique is defined.
                    - default: 'uniform'

        Input Shape:
            4-D tensor of shape (batch_size, inp_h, inp_w, inp_feature_maps)

        Output Shape:
            4-D tensor of shape (batch, out_h - np.sum(pad_h), out_w - np.sum(pad_w), self.filters)
            where,
                out_h = (inp_h - 1) * strides[0] + kernel_size[0]
                out_w = (inp_w - 1) * strides[1] + kernel_size[1]
                pad_h = total padding along height
                pad_w = total padding along width
            
            
    '''
    def __init__(self, filters, kernel_size, strides = (2, 2), padding = 'same',
                 activation = None, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        
        self.input_shape = None
        self.all_weights = ()

        self.name = kwargs.get('name', self.__class__.__name__)
        self.trainable = kwargs.get('trainable', True)
        self.weight_initializer = kwargs.get('weight_initializer', 'uniform')


    def preset_optimizer(self, optimizer):
        self.kernel_opt = copy(optimizer)
        self.bias_opt = copy(optimizer)

    def build(self, input_shape):
        self.input_shape = input_shape
        _, inp_h, inp_w, inp_chn = input_shape
        self.kernel = WeightsInitializer(shape = (self.kernel_size[0], self.kernel_size[1], self.filters, inp_chn),
                                         trainable = True, initializer = self.weight_initializer,
                                         name = f'{self.name}|Kernel:kernel')
        self.bias = WeightsInitializer(shape = (1, self.filters), trainable = True,
                                       initializer = 'zeros',
                                       name = f'{self.name}|Bias:bias')

        self.all_weights = (self.kernel, self.bias)


    def __call__(self, inputs):
        self.inputs = inputs
        if not hasattr(self, 'bias'):
            self.build(inputs.shape)

        batch_size, inp_h, inp_w, inp_chn = self.input_shape

        out_h = (inp_h - 1) * self.strides[0] + self.kernel_size[0]
        out_w = (inp_w - 1) * self.strides[1] + self.kernel_size[1]
        
        # fancy indexing
        i0 = np.repeat(np.arange(inp_h), np.prod(self.kernel_size)*inp_chn*inp_w).reshape(-1, np.prod(self.kernel_size)*inp_chn)\
                        .reshape(-1, np.prod(self.kernel_size), 1, inp_chn)
        j0 = np.tile(np.repeat(np.arange(inp_w), np.prod(self.kernel_size)*inp_chn).reshape(inp_w, -1).T, inp_h).T\
                    .reshape(-1, np.prod(self.kernel_size), 1, inp_chn)
        k0 = np.repeat(np.arange(inp_chn), np.prod(self.kernel_size)).reshape(inp_chn, np.prod(self.kernel_size)).T[:, np.newaxis, :]
        #####

        new_inp = self.inputs[:, i0, j0, k0]
        out_ = np.sum((new_inp * self.kernel.weight.reshape(np.prod(self.kernel_size), self.filters, inp_chn)\
                       [np.newaxis, np.newaxis, :, :, :]), axis = -1)


        # fancy indexing
        i1 = np.tile(np.tile(np.repeat(np.arange(self.kernel_size[0]), self.kernel_size[1]), inp_w*self.filters).reshape(inp_w, -1)\
        .reshape(inp_w, self.filters, np.prod(self.kernel_size)).T, inp_h).transpose(2, 0, 1) \
        + np.repeat(np.arange(0, self.strides[0]*inp_h, self.strides[0]), np.prod(self.kernel_size)*inp_w*self.filters)\
        .reshape(-1, np.prod(self.kernel_size)*self.filters).reshape(-1, np.prod(self.kernel_size), self.filters)

        j1 = np.tile(np.tile(np.tile(np.arange(self.kernel_size[1]), self.kernel_size[0]), inp_w * self.filters).reshape(inp_w, -1)\
            .reshape(inp_w, self.filters, np.prod(self.kernel_size)).T, inp_h).transpose(2, 0, 1)\
            + np.tile(np.repeat(np.arange(0, inp_w*self.strides[1], self.strides[1]), np.prod(self.kernel_size)*self.filters)\
            .reshape(-1, np.prod(self.kernel_size)*self.filters).reshape(inp_w, np.prod(self.kernel_size), self.filters).T, inp_h).T

        k1 = np.repeat(np.arange(self.filters), np.prod(self.kernel_size)).reshape(-1, np.prod(self.kernel_size)).T
        #####

        
        # Forward Propagation
        z = np.zeros((self.inputs.shape[0], out_h, out_w, self.filters))
        np.add.at(z, (slice(None), i1, j1, k1), out_)

        pad_h, pad_w = self._padding(self.kernel_size, self.strides, self.padding)

        if ((pad_h[0] == 0) & (pad_h[1] == 0)) | ((pad_w[0] == 0) & (pad_w[1] == 0)):
            pass
        else:
            zout = z[:, pad_h[0]:-pad_h[1], pad_w[0]:-pad_w[1], :]


        self.act = Activation(self.activation)
        zout = self.act(zout) if self.activation is not None else zout

        assert zout.shape[1:] == self.output_shape[1:]
        return zout


    def backprop(self, prev_dy):
        batch_size, inp_h, inp_w, inp_chn = self.input_shape
        pad_h, pad_w = self._padding(self.kernel_size, self.strides, self.padding)

        # BackPropagation
        prev_dy = self.act.backprop(prev_dy) if self.activation is not None else prev_dy 
        if self.trainable:
            _, prev_h, prev_w, _ = prev_dy.shape

            # fancy indexing
            h = np.tile(np.tile(np.repeat(np.arange(0, prev_h, self.strides[0]), inp_w).reshape(-1, 1),\
                    self.kernel_size[1]), self.kernel_size[0]).T \
                + np.repeat(np.arange(self.kernel_size[1]), self.kernel_size[0]*inp_h*inp_w)\
                .reshape(np.prod(self.kernel_size), -1)

            w = np.tile(np.tile(np.tile(np.arange(0, prev_w, self.strides[1]), inp_h).reshape(-1, 1), self.kernel_size[1]), self.kernel_size[0]).T\
                + np.tile(np.tile(np.arange(self.kernel_size[1]), self.kernel_size[0]*inp_w).reshape(-1, np.prod(self.kernel_size)).T, inp_h)
            #####

            prev_z_ = np.pad(prev_dy, ((0, 0), (pad_h[0], pad_h[1]), (pad_w[0], pad_w[1]), (0, 0)), mode = 'constant')
            prev_z = prev_z_[:, h, w, :].transpose(1, 3, 2, 0).reshape(np.prod(self.kernel_size), self.filters, -1)

            inp_col = self.inputs.reshape(-1, inp_chn)

            d_kernel = (prev_z @ inp_col).reshape(self.kernel_size[0], self.kernel_size[1], self.filters, inp_chn)
            d_bias = prev_z_.sum(axis = (0, 1, 2))[np.newaxis, :]

            # updating weights using an optimizer
            grad_pairs = [(self.kernel_opt, self.kernel, d_kernel),
                          (self.bias_opt, self.bias, d_bias)]
            self._update_weights(grad_pairs)

        
        z = np.pad(prev_dy, ((0, 0), (pad_h[0], pad_h[1]), (pad_w[0], pad_w[1]), (0, 0)),
                   mode = 'constant')

        # fancy indexing
        i = np.tile(np.tile(np.tile(np.repeat(np.arange(self.kernel_size[0]), self.kernel_size[1]), self.filters), inp_w)\
                .reshape(inp_w, -1).T, inp_h).T \
                + np.repeat(np.arange(0, inp_h * self.strides[0], self.strides[0]), np.prod(self.kernel_size) * self.filters * inp_w)\
                .reshape(-1, np.prod(self.kernel_size) * self.filters)

        j = np.tile(np.tile(np.tile(np.arange(self.kernel_size[0]), self.kernel_size[1] * self.filters), inp_w)\
                .reshape(-1, np.prod(self.kernel_size) * self.filters).T, inp_h) \
                + np.tile(np.repeat(np.arange(0, inp_w * self.strides[1], self.strides[1]), np.prod(self.kernel_size) * self.filters)\
                .reshape(-1, np.prod(self.kernel_size) * self.filters).T, inp_h)

        k = np.repeat(np.arange(self.filters), np.prod(self.kernel_size)).reshape(1, -1)
        #####

        z_col = z[:, i, j.T, k]
        k_col = self.kernel.weight.reshape(np.prod(self.kernel_size), self.filters, inp_chn)\
                .transpose(1, 0, 2).reshape(-1, inp_chn)

        dx = (z_col @ k_col).reshape(self.inputs.shape[0], inp_h, inp_w, inp_chn)

        assert dx.shape == self.inputs.shape
        return dx


    @property
    def output_shape(self):
        batch, inp_h, inp_w, inp_chn = self.input_shape
        out_h = (inp_h - 1) * self.strides[0] + self.kernel_size[0]
        out_w = (inp_w - 1) * self.strides[1] + self.kernel_size[1]
        pad_h, pad_w = self._padding(self.kernel_size, self.strides, self.padding)

        return (batch, out_h - np.sum(pad_h), out_w - np.sum(pad_w), self.filters)




############################################################################################
                    #     CODE         ENDS         HERE     #
############################################################################################


            
