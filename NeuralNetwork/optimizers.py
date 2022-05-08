import numpy as np


# Optimizer
class Optimizer(object):
    '''
        Optimizer (Base Class)
            This is a parent class for all other optimizers.
    '''
    def __call__(self, grad_pairs):
        return NotImplementedError('Subclasses should implement this.')


# StochasticGradientDescent
class StochasticGradientDescent(Optimizer):
    '''
        Stochastic Gradient Descent (Base Class: Optimizer)
            This is a simple gradient descent with mini-batches. Momentum
            is also applied here.

        Args:
            learning_rate:
                - type: float
                - about: used for setting learning rate, which will be used
                         for optimizing weights.
                - default: 0.001

            momentum:
                - type: float
                - about: This value ranges between 0 and 1, it helps in accelerating
                         gradient descent and also in smooth decrease in loss.
                - default: 0.0

            grad_clip:
                - type: tuple
                - about: If not None, then the weights are clipped between
                         the range specified in the argument.
                - default: None

            normalize_grads:
                - type: bool
                - about: If True, then it will normalize the gradients.
                - default: False

            minimize:
                - type: bool
                - about: If True, it will reduce the loss by updating the gradient
                         in appropriate direction, if False, then it will increase
                         the loss by updating the loss in the opposite direction.
                - default: True
                        
    '''
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.0,
                 grad_clip: tuple = None, normalize_grads: bool = False,
                 minimize: bool = True):
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.grad_clip = grad_clip
        self.normalize_grads = normalize_grads
        self.minimize = minimize

    def __call__(self, weight, grad):
            
        if self.normalize_grads:
            grad /= np.std(grad) + 1e-12

        if self.grad_clip is not None:
            grad = np.clip(grad, self.grad_clip[0], self.grad_clip[1])

        if not hasattr(self, 'V'):
            self.V = np.zeros_like(grad)

        self.V = self.momentum * self.V + (1 - self.momentum) * grad

        if self.minimize:
            weight -= self.learning_rate * grad
        else:
            weight += self.learning_rate * grad



# AdaGrad
class Adagrad(Optimizer):
    '''
        AdaGrad (Base Class: Optimizer)
            Adagrad consists of adaptive learning rate, that is the learning rate
            as the training goes on and on.


        Args:
            learning_rate:
                - type: float
                - about: used for setting learning rate, which will be used
                         for optimizing weights.
                - default: 0.001

            grad_clip:
                - type: tuple
                - about: If not None, then the weights are clipped between
                         the range specified in the argument.
                - default: None

            normalize_grads:
                - type: bool
                - about: If True, then it will normalize the gradients.
                - default: False

            epsilon:
                - type: float
                - about: It is a very small number used to avoid, Zero Division Error.
                - default: 1e-8

            minimize:
                - type: bool
                - about: If True, it will reduce the loss by updating the gradient
                         in appropriate direction, if False, then it will increase
                         the loss by updating the loss in the opposite direction.
                - default: True
                
    '''
    def __init__(self, learning_rate: float = 0.001, grad_clip: tuple = None,
                 normalize_grads: bool = False, epsilon: float = 1e-8,
                 minimize: bool = True):
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.normalize_grads = normalize_grads
        self.epsilon = epsilon
        self.minimize = minimize

    def __call__(self, weight, grad):
            
        if self.normalize_grads:
            grad /= np.std(grad) + 1e-12

        if self.grad_clip is not None:
            grad = np.clip(grad, self.grad_clip[0], self.grad_clip[1])

        if not hasattr(self, 'alpha'):
            self.alpha = np.zeros_like(grad)

        self.alpha += np.square(grad)
        new_lr = self.learning_rate / np.sqrt(self.alpha + self.epsilon)

        if self.minimize:
            weight -= new_lr * grad
        else:
            weight += new_lr * grad



# RMSprop
class RMSprop(Optimizer):
    '''
        RMSprop (Base Class: Optimizer)
            RMSprop consists of adaptive learning rate, which based on the
            moving average of the square of gradients.


        Args:
            learning_rate:
                - type: float
                - about: used for setting learning rate, which will be used
                         for optimizing weights.
                - default: 0.001

            momentum:
                - type: float
                - about: This value ranges between 0 and 1, it helps in accelerating
                         gradient descent and also in smooth decrease in loss.
                - default: 0.0

            grad_clip:
                - type: tuple
                - about: If not None, then the weights are clipped between
                         the range specified in the argument.
                - default: None

            normalize_grads:
                - type: bool
                - about: If True, then it will normalize the gradients.
                - default: False

            epsilon:
                - type: float
                - about: It is a very small number used to avoid, Zero Division Error.
                - default: 1e-8

            minimize:
                - type: bool
                - about: If True, it will reduce the loss by updating the gradient
                         in appropriate direction, if False, then it will increase
                         the loss by updating the loss in the opposite direction.
                - default: True
                
    '''
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.0,
                 grad_clip: tuple = None, normalize_grads: bool = False,
                 epsilon: float = 1e-8, minimize: bool = True):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.grad_clip = grad_clip
        self.normalize_grads = normalize_grads
        self.epsilon = epsilon
        self.minimize = minimize

    def __call__(self, weight, grad):
            
        if self.normalize_grads:
            grad /= np.std(grad) + 1e-12

        if self.grad_clip is not None:
            grad = np.clip(grad, self.grad_clip[0], self.grad_clip[1])

        if not hasattr(self, 'S'):
            self.S = np.zeros_like(grad)

        self.S = self.momentum * self.S + (1 - self.momentum) * np.square(grad)
        new_lr = self.learning_rate / np.sqrt(self.S + self.epsilon)

        if self.minimize:
            weight -= new_lr * grad
        else:
            weight += new_lr * grad


# Adam
class Adam(Optimizer):
    '''
        Adam (Base Class: Optimizer)
            Adam consists of moving average of both gradients and learning rate.


        Args:
            learning_rate:
                - type: float
                - about: used for setting learning rate, which will be used
                         for optimizing weights.
                - default: 0.001

            beta_1:
                - type: float
                - about: This value ranges between 0 and 1, it helps in accelerating
                         gradient descent and also in smooth decrease in loss. It is used
                         for moving average of gradients.
                - default: 0.9

            beta_2:
                - type: float
                - about: This value ranges between 0 and 1, it helps in accelerating
                         gradient descent and also in smooth decrease in loss. It is used
                         for moving average of learning rate (adaptive).
                - default: 0.999

            grad_clip:
                - type: tuple
                - about: If not None, then the weights are clipped between
                         the range specified in the argument.
                - default: None

            normalize_grads:
                - type: bool
                - about: If True, then it will normalize the gradients.
                - default: False

            epsilon:
                - type: float
                - about: It is a very small number used to avoid, Zero Division Error.
                - default: 1e-12

            minimize:
                - type: bool
                - about: If True, it will reduce the loss by updating the gradient
                         in appropriate direction, if False, then it will increase
                         the loss by updating the loss in the opposite direction.
                - default: True
                
    '''
    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9,
                 beta_2: float = 0.999, grad_clip: tuple = None,
                 normalize_grads: bool = False, epsilon: float = 1e-12,
                 minimize: bool = True):
        
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.grad_clip = grad_clip
        self.normalize_grads = normalize_grads
        self.epsilon = epsilon
        self.minimize = minimize

    def __call__(self, weight, grad):
            
        if self.normalize_grads:
            grad /= np.std(grad) + 1e-12

        if self.grad_clip is not None:
            grad = np.clip(grad, self.grad_clip[0], self.grad_clip[1])

        if not hasattr(self, 'S'):
            self.S = np.zeros_like(grad)
        if not hasattr(self, 'V'):
            self.V = np.zeros_like(grad)

        self.V = self.beta_1 * self.V + (1 - self.beta_1) * grad
        self.S = self.beta_2 * self.S + (1 - self.beta_2) * np.square(grad)

        v_hat = self.V / (1 - self.beta_1)
        s_hat = self.S / (1 - self.beta_2)

        if self.minimize:
            weight -= self.learning_rate * (v_hat / np.sqrt(s_hat + self.epsilon))
        else:
            weight += self.learning_rate * (v_hat / np.sqrt(s_hat + self.epsilon))

     

        
