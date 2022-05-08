import numpy as np

# Loss
class Loss(object):
    '''
        Loss (Base Class)
            This is the parent class to all loss function classes.
    '''
    def __call__(self, y_true, y_pred):
        raise NotImplementedError('Subclasses should implement this.')
    
    def backprop(self):
        raise NotImplementedError('Subclasses should implement this.')


# MeanSquareError
class MeanSquaredError(Loss):
    '''
        MeanSquaredError (Base Class: Loss)

        - Mainly used for regression task.
    '''
    def __call__(self, y_true, y_pred):
        self.y_true, self.y_pred = y_true, y_pred
        return np.mean(np.square(y_true - y_pred))

    def backprop(self):
        return -2 * np.mean(self.y_true - self.y_pred)


# MeanAbsoluteError
class MeanAbsoluteError(Loss):
    '''
        MeanAbsoluteError (Base Class: Loss)

        - Mainly usef for regression task.
    '''
    def __call__(self, y_true, y_pred):
        self.y_true, self.y_pred = y_true, y_pred
        return np.mean(np.abs(y_true - y_pred))

    def backprop(self):
        return np.where(self.y_pred > self.y_true, 1, -1)


# BinaryCrossEntropy
class BinaryCrossEntropy(Loss):
    '''
        BinaryCrossEntropy (Base Class: Loss)

        - Mainly used for binary classification task.
    '''
    def __call__(self, y_true, y_pred):
        self.y_true, self.y_pred = y_true, y_pred
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backprop(self):
        return -self.y_true/self.y_pred + (1 - self.y_true)/(1 - self.y_pred)


# CategoricalCrossEntropy
class CategoricalCrossEntropy(Loss):
    '''
        CategoricalCrossEntropy (Base Class: Loss)

        - Mainly used for multi-class classification task.
    '''
    def __call__(self, y_true, y_pred):
        self.y_true, self.y_pred = y_true, y_pred
        #print(y_true, y_pred)
        return -np.sum(y_true * np.log(y_pred))

    def backprop(self):
        return -self.y_true/self.y_pred


# SparseCategoricalCrossEntropy
class SparseCategoricalCrossEntropy(Loss):
    '''
        SparseCategoricalCrossEntropy (Base Class: Loss)

        - Can be used for both binary & multi-class classification.
          Input is given sparsely.
    '''
    def __init__(self):
        self.cce = CategoricalCrossEntropy()

    def __call__(self, y_true, y_pred):
        y_uniq = np.unique(y_true)
        yt = np.zeros((y_true.shape[0], len(y_uniq)))

        for n, i in enumerate(y_true):
            yt[n, i] = 1

        return self.cce(yt, y_pred)

    def backprop(self):
        return self.cce.backprop()


